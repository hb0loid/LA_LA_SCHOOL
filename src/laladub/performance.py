from __future__ import annotations

import json
import math
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_WRITE_LOCK = threading.Lock()
_MIN_SAMPLE_SECONDS = 10.0
_MAX_SAMPLE_SECONDS = 8 * 60 * 60.0


@dataclass(frozen=True, slots=True)
class RuntimeEstimate:
    seconds: float
    low_seconds: float
    high_seconds: float
    sample_count: int


def _number(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def job_duration_seconds(job: dict[str, Any]) -> float | None:
    for key, divisor in (
        ("quota_duration_ms", 1000.0),
        ("daily_trimmed_duration_ms", 1000.0),
        ("input_duration_ms", 1000.0),
        ("input_duration_seconds", 1.0),
        ("duration", 1.0),
    ):
        value = _number(job.get(key))
        if value is not None and value > 0:
            return value / divisor
    return None


def _stage_seconds(job: dict[str, Any]) -> dict[str, float]:
    raw = job.get("stage_seconds")
    if not isinstance(raw, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in raw.items():
        seconds = _number(value)
        if seconds is not None and seconds >= 0:
            result[str(key)[:120]] = round(seconds, 3)
    return result


def performance_record(job_dir: Path, job: dict[str, Any]) -> dict[str, Any] | None:
    """Build a content-free performance row from a terminal job snapshot."""
    status = str(job.get("status") or "")
    if status not in {"done", "failed", "rejected"}:
        return None
    queued_at = _number(job.get("first_queued_at")) or _number(job.get("queued_at"))
    started_at = _number(job.get("started_at"))
    finished_at = _number(job.get("finished_at")) or time.time()
    # Older recovery code overwrote queued_at after started_at. Do not turn
    # that into negative queue time or discard the remote preprocessing span.
    if queued_at is not None and started_at is not None and queued_at > started_at:
        queued_at = started_at
    total_seconds = finished_at - queued_at if queued_at is not None else None
    processing_seconds = finished_at - started_at if started_at is not None else None
    if total_seconds is not None and total_seconds < 0:
        total_seconds = None
    if processing_seconds is not None and processing_seconds < 0:
        processing_seconds = None
    error_text = str(job.get("error") or "").strip()
    return {
        "schema": 1,
        "event_id": f"{job_dir.parent.name}/{job_dir.name}:{finished_at:.3f}:{status}",
        "recorded_at": time.time(),
        "job_number": job_dir.name,
        "status": status,
        "mode": str(job.get("mode") or "dub"),
        "duration_seconds": job_duration_seconds(job),
        "queue_seconds": (
            max(0.0, started_at - queued_at)
            if queued_at is not None and started_at is not None
            else None
        ),
        "processing_seconds": processing_seconds,
        "total_seconds": total_seconds,
        "tts": str(job.get("tts_provider") or ""),
        "asr_method": str(job.get("asr_method") or ""),
        "source_lang": str(job.get("source_lang") or "auto"),
        "target_lang": str(job.get("target_lang") or ""),
        "queue_priority": str(job.get("queue_priority") or ""),
        "queue_attempt_at": _number(job.get("queue_attempt_at")),
        "remote_preprocess": bool(job.get("remote_preprocess_completed_at")),
        "remote_preprocess_seconds": _number(job.get("remote_preprocess_seconds")),
        "remote_preprocess_worker": str(job.get("remote_preprocess_worker") or ""),
        "local_continuation_seconds": _number(job.get("local_continuation_seconds")),
        "stage_seconds": _stage_seconds(job),
        "initial_eta_seconds": _number(job.get("initial_eta_seconds")),
        "initial_eta_samples": int(_number(job.get("initial_eta_samples")) or 0),
        "error_type": error_text.split(":", 1)[0][:120] if error_text else "",
    }


def record_terminal_job(job_dir: Path, job: dict[str, Any]) -> None:
    """Append one durable metrics row before retention removes the job folder."""
    record = performance_record(job_dir, job)
    if record is None:
        return
    marker = job_dir / ".performance-recorded"
    telemetry_dir = job_dir.parent.parent / "_telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    path = telemetry_dir / "performance.jsonl"
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _WRITE_LOCK:
        try:
            recorded_ids = set(marker.read_text(encoding="utf-8").splitlines())
        except OSError:
            recorded_ids = set()
        event_id = str(record["event_id"])
        if event_id in recorded_ids:
            return
        with path.open("a", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
        with marker.open("a", encoding="utf-8") as stream:
            stream.write(event_id + "\n")


class PerformanceHistory:
    """Robust local runtime model used for diagnostics and approximate ETAs."""

    def __init__(self, workdir: Path, *, refresh_seconds: float = 60.0) -> None:
        self.workdir = workdir
        self.metrics_path = workdir / "_telemetry" / "performance.jsonl"
        self.refresh_seconds = refresh_seconds
        self._last_refresh = 0.0
        self._samples: list[dict[str, Any]] = []
        self.refresh(force=True)

    @property
    def sample_count(self) -> int:
        self.refresh()
        return len(self._samples)

    def refresh(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_refresh < self.refresh_seconds:
            return
        self._last_refresh = now
        records = list(self._read_jsonl())
        if len(records) < 100:
            bootstrap = self._bootstrap_job_snapshots(limit=500)
            records.extend(bootstrap)
            self._persist_bootstrap(bootstrap, known_records=records)
        unique: dict[str, dict[str, Any]] = {}
        for record in records:
            event_id = str(record.get("event_id") or "")
            if event_id:
                unique[event_id] = record
        self._samples = [sample for sample in unique.values() if self._valid_sample(sample)]

    def _persist_bootstrap(
        self,
        records: list[dict[str, Any]],
        *,
        known_records: list[dict[str, Any]],
    ) -> None:
        if not records:
            return
        # The first run turns recent job.json history into durable, tiny rows so
        # the normal media-retention cleanup cannot erase the ETA baseline.
        existing_count = max(0, len(known_records) - len(records))
        existing_ids = {
            str(record.get("event_id") or "")
            for record in known_records[:existing_count]
            if record.get("event_id")
        }
        missing = [record for record in records if str(record.get("event_id") or "") not in existing_ids]
        if not missing:
            return
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK:
            with self.metrics_path.open("a", encoding="utf-8") as stream:
                for record in missing:
                    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def estimate(self, job: dict[str, Any]) -> RuntimeEstimate | None:
        self.refresh()
        duration = job_duration_seconds(job)
        if duration is None or not self._samples:
            return None
        mode = str(job.get("mode") or "dub")
        tts = str(job.get("tts_provider") or "")
        candidates = [
            sample
            for sample in self._samples
            if str(sample.get("mode") or "dub") == mode
            and (mode != "dub" or not tts or str(sample.get("tts") or "") == tts)
        ]
        if len(candidates) < 5:
            candidates = [sample for sample in self._samples if str(sample.get("mode") or "dub") == mode]
        if len(candidates) < 5:
            candidates = list(self._samples)
        predictions = self._scaled_predictions(candidates, duration)
        if len(predictions) < 3:
            return None
        predictions.sort()
        median = statistics.median(predictions)
        low = self._percentile(predictions, 0.25) * 0.85
        high = self._percentile(predictions, 0.75) * 1.20
        return RuntimeEstimate(
            seconds=max(30.0, median),
            low_seconds=max(15.0, min(low, median)),
            high_seconds=max(median, high),
            sample_count=len(predictions),
        )

    def _read_jsonl(self) -> Iterable[dict[str, Any]]:
        if not self.metrics_path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            with self.metrics_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        value = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(value, dict):
                        records.append(value)
        except OSError:
            return []
        return records[-2000:]

    def _bootstrap_job_snapshots(self, *, limit: int) -> list[dict[str, Any]]:
        try:
            paths = sorted(
                self.workdir.glob("*/*/job.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )[:limit]
        except OSError:
            return []
        result: list[dict[str, Any]] = []
        for path in paths:
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                continue
            if not isinstance(job, dict):
                continue
            record = performance_record(path.parent, job)
            if record is not None:
                result.append(record)
        return result

    @staticmethod
    def _valid_sample(sample: dict[str, Any]) -> bool:
        if str(sample.get("status") or "") != "done":
            return False
        duration = _number(sample.get("duration_seconds"))
        total = _number(sample.get("total_seconds"))
        return bool(
            duration is not None
            and duration > 0
            and total is not None
            and _MIN_SAMPLE_SECONDS <= total <= _MAX_SAMPLE_SECONDS
        )

    @staticmethod
    def _scaled_predictions(samples: list[dict[str, Any]], target_duration: float) -> list[float]:
        # Runtime has a large fixed model-loading cost, so scale sub-linearly.
        # Keep nearby duration buckets to avoid a 10-second clip predicting a
        # ten-minute video's runtime (or the reverse).
        nearby: list[dict[str, Any]] = []
        for sample in samples:
            duration = _number(sample.get("duration_seconds")) or 0.0
            ratio = duration / max(target_duration, 1.0)
            if 0.25 <= ratio <= 4.0:
                nearby.append(sample)
        selected = nearby if len(nearby) >= 5 else samples
        predictions: list[float] = []
        for sample in selected[-500:]:
            duration = _number(sample.get("duration_seconds"))
            total = _number(sample.get("total_seconds"))
            if duration is None or total is None or duration <= 0:
                continue
            predictions.append(total * (target_duration / duration) ** 0.65)
        if len(predictions) >= 8:
            predictions.sort()
            trim = max(1, len(predictions) // 10)
            predictions = predictions[trim:-trim]
        return predictions

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        index = max(0, min(len(values) - 1, round((len(values) - 1) * fraction)))
        return values[index]


def merge_stage_seconds(existing: object, incoming: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(existing, dict):
        for key, value in existing.items():
            seconds = _number(value)
            if seconds is not None and seconds >= 0:
                result[str(key)] = seconds
    for key, value in incoming.items():
        result[key] = round(result.get(key, 0.0) + max(0.0, float(value)), 3)
    return result
