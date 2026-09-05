from __future__ import annotations

import argparse
import contextlib
import getpass
import http.client
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

from .bot_config import BotSettings, load_bot_settings
from .job_runner import execute_job, result_manifest, save_job_snapshot


DEFAULT_CONFIG_PATH = Path("worker_config.json")
LOCAL_VERSION_PATH = Path("worker_version.json")
LOCAL_STATE_PATH = Path(".worker_state.json")
UPDATE_EXIT_CODE = 42


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config_path = args.config
    file_config = _load_worker_config(config_path)

    server = args.server or os.environ.get("LALADUB_WORKER_SERVER") or file_config.get("server") or "http://127.0.0.1:8765"
    token = args.token or os.environ.get("LALADUB_WORKER_TOKEN") or file_config.get("token") or ""
    worker_id = args.worker_id or os.environ.get("LALADUB_WORKER_ID") or file_config.get("worker_id") or _default_worker_id()
    poll_seconds = max(1.0, args.poll_seconds)
    workdir = args.workdir

    if not token and sys.stdin.isatty():
        token = getpass.getpass("Worker token: ").strip()
    if not token:
        raise SystemExit("Worker token is required. Set LALADUB_WORKER_TOKEN or put it in worker_config.json.")

    if not config_path.exists() and not args.no_save_config:
        _save_worker_config(config_path, {"server": server, "token": token, "worker_id": worker_id})

    client = CoordinatorClient(server, token, worker_id)
    token = _rotate_worker_token(
        client,
        config_path=config_path,
        file_config=file_config,
        server=server,
        worker_id=worker_id,
        save_config=not args.no_save_config,
    )
    workdir.mkdir(parents=True, exist_ok=True)
    client.contact_path = _start_stall_heartbeat(workdir)
    _ensure_windows_autostart()
    print(f"LaLaDub worker started: id={worker_id} server={server} workdir={workdir}", flush=True)
    try:
        _report_startup_logs(client, workdir)
    except Exception as exc:
        print(f"Startup log report failed: {type(exc).__name__}: {exc}", flush=True)

    if args.auto_update and _remote_update_available(client):
        print("Worker update is available. Restarting through launcher.", flush=True)
        raise SystemExit(UPDATE_EXIT_CODE)

    while True:
        try:
            lease = client.lease(worker_id)
            if lease is None:
                if args.auto_update and _remote_update_available(client):
                    print("Worker update is available. Restarting through launcher.", flush=True)
                    raise SystemExit(UPDATE_EXIT_CODE)
                if args.once:
                    print("No job available.", flush=True)
                    return
                time.sleep(poll_seconds)
                continue
            _run_lease(client, lease, workdir)
        except KeyboardInterrupt:
            print("Worker stopped by user.", flush=True)
            return
        except Exception:
            print(traceback.format_exc(), flush=True)
            if args.once:
                raise
            time.sleep(poll_seconds)


# Enough to ride out a router hiccup without keeping a dead link alive for long.
UPLOAD_ATTEMPTS = 4
UPLOAD_RETRY_SECONDS = 5.0
# Long enough that the lookup leaves the hot path entirely, short enough that a
# machine given a new address on the LAN is found again without a restart.
HOST_CACHE_SECONDS = 600.0


class CoordinatorClient:
    def __init__(self, server: str, token: str, worker_id: str = "worker") -> None:
        self.server = server.rstrip("/")
        self.token = token
        # Sent with every progress and heartbeat post. The coordinator needs it
        # to tell a worker is alive even when the job it reports on is one the
        # coordinator has already taken back.
        self.worker_id = str(worker_id or "worker")
        # Set once the workdir is known. Every successful request stamps it, and
        # the supervisor kills a worker whose stamp goes stale - so a request
        # wedged in a name lookup is caught, which mere process liveness missed.
        self.contact_path: Path | None = None
        self._contact_written = 0.0
        # The coordinator is configured by name, and a name lookup in Python has
        # no timeout and is not covered by the socket timeout. Doing one per
        # request is what let a single wedged lookup silence the worker for
        # thirty hours. Resolve once, reuse, and re-resolve only when a request
        # fails - which is also how a changed address recovers.
        self._resolved_host: str | None = None
        self._resolved_at = 0.0
        self._parsed = urllib.parse.urlparse(self.server)
        if self._parsed.scheme not in {"http", "https"}:
            raise ValueError("Worker server must be http:// or https:// URL.")

    def lease(self, worker_id: str) -> dict[str, Any] | None:
        path = f"/api/v1/jobs/lease?worker_id={urllib.parse.quote(worker_id)}"
        response = self._request_json("GET", path, none_on_204=True)
        if response is None:
            return None
        if not isinstance(response, dict):
            raise RuntimeError(f"Bad lease response: {response!r}")
        return response

    def send_startup_report(self, sections: dict[str, str]) -> None:
        """Hands the coordinator the tail of this worker's own logs.

        The link only ever runs one way - the laptop dials the main PC, and
        nothing on the laptop answers from outside. So when a worker vanishes,
        the reason is written down on a machine nobody can read from the other
        side. Sending it on the way back up closes that gap.
        """
        self._request_json(
            "POST",
            "/api/v1/worker/report",
            {"worker_id": self.worker_id, "sections": sections},
            timeout=30,
        )

    def worker_manifest(self) -> dict[str, Any] | None:
        response = self._request_json("GET", "/api/v1/worker/manifest")
        if response is None:
            return None
        if not isinstance(response, dict):
            raise RuntimeError(f"Bad worker manifest response: {response!r}")
        return response

    def download_input(self, job_id: str, destination: Path) -> None:
        url = self._url(f"/api/v1/jobs/{urllib.parse.quote(job_id)}/input")
        request = urllib.request.Request(url, headers=self._auth_headers())
        with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)

    def progress(self, job_id: str, stage: str, current: int | None, total: int | None, detail: str | None) -> None:
        self._request_json(
            "POST",
            f"/api/v1/jobs/{urllib.parse.quote(job_id)}/progress",
            {
                "stage": stage,
                "current": current,
                "total": total,
                "detail": detail,
                "worker_id": self.worker_id,
            },
        )

    def heartbeat(self, job_id: str) -> None:
        # A short timeout on purpose. The coordinator drops the lease after a
        # fixed silence; a single call blocking on the default timeout outlasts
        # that window, so the job is taken away from a worker still running it.
        self._request_json(
            "POST",
            f"/api/v1/jobs/{urllib.parse.quote(job_id)}/progress",
            {"heartbeat_only": True, "worker_id": self.worker_id},
            timeout=20,
        )

    def upload_file(self, job_id: str, kind: str, path: Path) -> None:
        query = urllib.parse.urlencode({"filename": path.name})
        request_path = f"/api/v1/jobs/{urllib.parse.quote(job_id)}/result/{urllib.parse.quote(kind)}?{query}"
        # This carries the whole job home. One dropped connection used to throw
        # away everything the laptop had just spent twenty minutes computing,
        # and the job was then redone from scratch on the main PC. The upload
        # is a PUT to a fixed path, so repeating it is safe.
        for attempt in range(UPLOAD_ATTEMPTS):
            try:
                self._upload_file(request_path, path)
                return
            except (OSError, http.client.HTTPException) as exc:
                if attempt == UPLOAD_ATTEMPTS - 1:
                    raise
                print(
                    f"Result upload failed ({type(exc).__name__}: {exc}); "
                    f"retrying {attempt + 2}/{UPLOAD_ATTEMPTS}",
                    flush=True,
                )
                time.sleep(UPLOAD_RETRY_SECONDS * (attempt + 1))

    def complete(self, job_id: str, manifest: dict[str, Any]) -> None:
        self._request_json("POST", f"/api/v1/jobs/{urllib.parse.quote(job_id)}/complete", manifest)

    def fail(self, job_id: str, error: str, traceback_text: str) -> None:
        self._request_json(
            "POST",
            f"/api/v1/jobs/{urllib.parse.quote(job_id)}/fail",
            {"error": error, "traceback": traceback_text},
        )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        none_on_204: bool = False,
        timeout: float = 120,
    ) -> Any:
        data = None
        headers = self._auth_headers()
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        request = urllib.request.Request(self._url(path), data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status == 204 and none_on_204:
                    self.note_contact()
                    return None
                body = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Coordinator HTTP {exc.code}: {body}") from exc
        except OSError:
            # Reaching the cached address failed; look it up again next time.
            self.forget_host()
            raise
        self.note_contact()
        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    def _upload_file(self, path: str, file_path: Path) -> None:
        connection_cls = http.client.HTTPSConnection if self._parsed.scheme == "https" else http.client.HTTPConnection
        host = self._connect_host()
        port = self._parsed.port
        connection = connection_cls(host, port, timeout=300)
        request_path = (self._parsed.path.rstrip("/") if self._parsed.path else "") + path
        headers = self._auth_headers()
        headers["Content-Type"] = "application/octet-stream"
        headers["Content-Length"] = str(file_path.stat().st_size)
        try:
            connection.putrequest("PUT", request_path)
            for name, value in headers.items():
                connection.putheader(name, value)
            connection.endheaders()
            with file_path.open("rb") as input_file:
                while True:
                    chunk = input_file.read(1024 * 1024)
                    if not chunk:
                        break
                    connection.send(chunk)
                    # Sending is talking. Without this the stamp would depend
                    # entirely on the lease heartbeat thread, and the stall
                    # limit had to be set wide enough to cover a whole upload.
                    self.note_contact()
            response = connection.getresponse()
            body = response.read().decode("utf-8", errors="replace")
            if response.status >= 300:
                raise RuntimeError(f"Coordinator upload HTTP {response.status}: {body}")
        finally:
            connection.close()

    def note_contact(self) -> None:
        path = self.contact_path
        if path is None:
            return
        now = time.time()
        # Once every few seconds is plenty; the supervisor looks every twenty.
        if now - self._contact_written < 5.0:
            return
        self._contact_written = now
        try:
            path.write_text(str(now), encoding="utf-8")
        except Exception:
            pass

    def _connect_host(self) -> str:
        """The address to dial, preferring a cached one over a fresh lookup."""
        host = self._parsed.hostname or "127.0.0.1"
        try:
            ipaddress.ip_address(host)
            return host
        except ValueError:
            pass
        now = time.time()
        if self._resolved_host and now - self._resolved_at < HOST_CACHE_SECONDS:
            return self._resolved_host
        try:
            info = socket.getaddrinfo(host, self._parsed.port, proto=socket.IPPROTO_TCP)
        except OSError:
            # Keep using whatever worked last; the name may come back.
            return self._resolved_host or host
        address = str(info[0][4][0])
        if address != self._resolved_host:
            print(f"Coordinator {host} resolved to {address}", flush=True)
        self._resolved_host = address
        self._resolved_at = now
        return address

    def forget_host(self) -> None:
        """Drops the cached address so the next attempt looks it up again."""
        self._resolved_at = 0.0

    def _netloc(self) -> str:
        host = self._connect_host()
        port = self._parsed.port
        return f"{host}:{port}" if port else host

    def _url(self, path: str) -> str:
        base_path = self._parsed.path.rstrip("/")
        return urllib.parse.urlunparse(
            (
                self._parsed.scheme,
                self._netloc(),
                f"{base_path}{path}",
                "",
                "",
                "",
            )
        )

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


class ProgressReporter:
    def __init__(self, client: CoordinatorClient, job_id: str, min_interval: float = 1.5) -> None:
        self.client = client
        self.job_id = job_id
        self.min_interval = min_interval
        self._last_sent = 0.0
        self._last_payload: tuple[str, int | None, int | None, str | None] | None = None

    def __call__(self, stage: str, current: int | None, total: int | None, detail: str | None) -> None:
        payload = (stage, current, total, detail)
        now = time.monotonic()
        if payload == self._last_payload and now - self._last_sent < self.min_interval:
            return
        if now - self._last_sent < self.min_interval and current not in {0, 100}:
            return
        self._last_payload = payload
        self._last_sent = now
        try:
            self.client.progress(self.job_id, stage, current, total, detail)
        except Exception as exc:
            print(f"Progress upload failed: {type(exc).__name__}: {exc}", flush=True)


def _run_lease(client: CoordinatorClient, lease: dict[str, Any], workdir: Path) -> None:
    job_id = str(lease["job_id"])
    remote_job = dict(lease.get("job") or {})
    input_filename = str(lease.get("input_filename") or "input.mp4")
    job_dir = workdir / "jobs" / _safe_name(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / _safe_name(input_filename)
    print(f"Leased job {job_id}: downloading {input_filename}", flush=True)

    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_lease_heartbeat_loop,
        args=(client, job_id, heartbeat_stop),
        name=f"laladub-heartbeat-{_safe_name(job_id)}",
        daemon=True,
    )
    heartbeat_thread.start()
    heartbeat_process = _start_external_lease_heartbeat(client, job_id)

    try:
        client.download_input(job_id, input_path)
        job = dict(remote_job)
        job["job_dir"] = str(job_dir)
        job["input_path"] = str(input_path)
        save_job_snapshot(job_dir, job, status="running")
        settings = _settings_for_worker_job(load_bot_settings(require_token=False), job)
        if settings.artifact_whisper_device == "cuda" and str(job.get("remote_stage") or "").strip().lower() == "preprocess":
            print("Worker preprocessing acceleration: artifact Whisper uses CUDA.", flush=True)
        reporter = ProgressReporter(client, job_id)
        reporter("Worker started", 1, 100, f"input={job.get('source_lang') or 'auto'}")
        result = execute_job(job, settings, progress_callback=reporter)
        manifest = result_manifest(result)
        _upload_result_files(client, job_id, result)
        client.complete(job_id, manifest)
        save_job_snapshot(job_dir, job, status="done")
        print(f"Completed job {job_id}", flush=True)
    except Exception as exc:
        traceback_text = traceback.format_exc()
        print(traceback_text, flush=True)
        save_job_snapshot(job_dir, remote_job, status="failed", error=str(exc))
        client.fail(job_id, "".join(traceback.format_exception_only(type(exc), exc)).strip(), traceback_text)
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2.0)
        _stop_external_lease_heartbeat(heartbeat_process)


def _settings_for_worker_job(settings: BotSettings, job: dict[str, Any]) -> BotSettings:
    if str(job.get("remote_stage") or "").strip().lower() != "preprocess" or not _cuda_available():
        return settings
    return replace(settings, artifact_whisper_device="cuda")


# Enough to cover a crash and the minutes before it, small enough to sit in a
# log line without drowning everything else in it.
REPORT_TAIL_BYTES = 6000


def _log_tail(path: Path, limit: int = REPORT_TAIL_BYTES) -> str:
    """The tail of a log file, whatever encoding it was written in.

    Windows PowerShell 5.1 redirects native output as UTF-16, so the first
    report that arrived read as text with a space between every letter. Which of
    the two a given file is depends on how its lines got there, so sniff rather
    than assume.
    """
    wide = False
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            wide = handle.read(2) == b"\xff\xfe"
            start = max(0, size - limit)
            if wide and start % 2:
                # Landing mid-character turns the whole tail into nonsense.
                start += 1
            handle.seek(start)
            data = handle.read()
    except Exception:
        return ""
    if not wide:
        wide = data.count(0) > len(data) // 4
    text = data.decode("utf-16-le" if wide else "utf-8", errors="replace")
    text = text.replace("﻿", "").replace("\x00", "").strip()
    if size > limit:
        # The first line is half a line; drop it rather than report a fragment.
        text = text.split("\n", 1)[-1]
    return text


def _worker_log_dir(workdir: Path) -> Path:
    """Where the launcher keeps worker.log and worker-supervisor.log.

    The supervisor runs the worker from the package root and passes
    runs/worker as the workdir, so the logs sit two levels up from it. Falling
    back to the current directory covers a worker started by hand.
    """
    candidates = [workdir.parent.parent / "logs", Path.cwd() / "logs"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[-1]


def _report_startup_logs(client: CoordinatorClient, workdir: Path) -> None:
    log_dir = _worker_log_dir(workdir)
    sections = {}
    for name in ("worker-supervisor.log", "worker.log", "worker.run.err.log"):
        tail = _log_tail(log_dir / name)
        if tail:
            sections[name] = tail
    if not sections:
        return
    client.send_startup_report(sections)


def _start_stall_heartbeat(workdir: Path) -> Path:
    """The file the supervisor watches to tell a hang from a long job.

    It marks the last time this worker *successfully reached the coordinator* -
    not merely the last time the process was alive. That distinction is the
    whole point: the worker talks to the main PC by name, and a name lookup has
    no timeout and does not hold the interpreter, so a lookup that never returns
    left every thread happily running while the worker said nothing to anyone
    for thirty hours. A watchdog keyed on liveness saw a healthy process and let
    it sit there.

    Touched from a thread only until the first real request lands, so a worker
    that never reaches the coordinator at all still gets replaced.
    """
    path = workdir / "worker_heartbeat.txt"
    try:
        path.write_text(str(time.time()), encoding="utf-8")
    except Exception:
        pass
    return path


def _lease_heartbeat_loop(client: CoordinatorClient, job_id: str, stop: threading.Event) -> None:
    while not stop.wait(10.0):
        try:
            client.heartbeat(job_id)
        except Exception as exc:
            print(f"Worker heartbeat failed: {type(exc).__name__}: {exc}", flush=True)


def _start_external_lease_heartbeat(client: CoordinatorClient, job_id: str) -> subprocess.Popen[bytes] | None:
    """Keep the lease alive even when CUDA code holds the worker's Python GIL.

    Some Whisper/model calls can prevent every thread in this interpreter from
    running for several minutes.  A heartbeat thread therefore cannot prove the
    machine disappeared.  A tiny sibling process is scheduled independently by
    Windows and keeps network liveness separate from the heavy ML process.
    """
    env = os.environ.copy()
    env.update(
        {
            "LALADUB_HEARTBEAT_SERVER": client.server,
            "LALADUB_HEARTBEAT_TOKEN": client.token,
            "LALADUB_HEARTBEAT_WORKER_ID": client.worker_id,
            "LALADUB_HEARTBEAT_JOB_ID": job_id,
        }
    )
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "laladub.worker_heartbeat"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            creationflags=creationflags,
        )
    except Exception as exc:
        # The in-process thread remains as a compatibility fallback.
        print(f"External heartbeat launch failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def _stop_external_lease_heartbeat(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    with contextlib.suppress(Exception):
        process.terminate()
        process.wait(timeout=3.0)
    if process.poll() is None:
        with contextlib.suppress(Exception):
            process.kill()


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _upload_result_files(client: CoordinatorClient, job_id: str, result: Any) -> None:
    if result.video_path is not None:
        client.upload_file(job_id, "video", result.video_path)
    if result.transcript_path is not None:
        client.upload_file(job_id, "transcript", result.transcript_path)
    for document in result.documents:
        client.upload_file(job_id, "documents", document.path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laladub-worker", description="La La Dub distributed worker.")
    parser.add_argument("--server", default=None, help="Coordinator URL, e.g. http://MAIN_PC_IP:8765")
    parser.add_argument("--token", default=None, help="Worker API token.")
    parser.add_argument("--worker-id", default=None, help="Human readable worker id.")
    parser.add_argument("--workdir", type=Path, default=Path("runs/worker"), help="Local worker jobs directory.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Worker config JSON path.")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true", help="Run one lease attempt and exit.")
    parser.add_argument("--no-auto-update", dest="auto_update", action="store_false", help="Disable idle update checks.")
    parser.add_argument("--no-save-config", action="store_true", help="Do not write worker_config.json on first run.")
    parser.set_defaults(auto_update=True)
    return parser


def _load_worker_config(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items() if value is not None}


def _save_worker_config(path: Path, data: dict[str, str]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _rotate_worker_token(
    client: CoordinatorClient,
    *,
    config_path: Path,
    file_config: dict[str, str],
    server: str,
    worker_id: str,
    save_config: bool,
) -> str:
    try:
        manifest = client.worker_manifest()
    except Exception as exc:
        print(f"Worker token sync skipped: {type(exc).__name__}: {exc}", flush=True)
        return client.token
    new_token = str((manifest or {}).get("worker_token") or "").strip() or client.token
    canonical_server = str((manifest or {}).get("worker_server") or "").strip().rstrip("/")
    saved_server = canonical_server if _server_health_available(canonical_server) else server
    token_changed = new_token != client.token
    server_changed = saved_server.rstrip("/") != server.rstrip("/")
    if not token_changed and not server_changed:
        return client.token

    client.token = new_token
    if save_config:
        updated = dict(file_config)
        updated.update({"server": saved_server, "token": new_token, "worker_id": worker_id})
        _save_worker_config(config_path, updated)
    if token_changed:
        print("Worker authentication token synchronized.", flush=True)
    if server_changed:
        print(f"Worker server switched to stable hostname: {saved_server}", flush=True)
    return new_token


def _server_health_available(server: str) -> bool:
    if not server:
        return False
    try:
        parsed = urllib.parse.urlparse(server)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False
        request = urllib.request.Request(f"{server.rstrip('/')}/health")
        with urllib.request.urlopen(request, timeout=5) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def _remote_update_available(client: CoordinatorClient) -> bool:
    try:
        manifest = client.worker_manifest()
    except Exception as exc:
        print(f"Update check skipped: {type(exc).__name__}: {exc}", flush=True)
        return False
    if not manifest or not manifest.get("available"):
        return False
    remote_build = str(manifest.get("build_id") or manifest.get("sha256") or "").strip()
    if not remote_build:
        return False
    local_build = _local_build_id()
    if not local_build:
        # An unknown build used to mean "no update", which is backwards: a
        # worker that cannot say what it is running is the one most in need of
        # replacing. The launcher already installs in this case, but it only
        # gets the chance once the worker asks to restart - so refusing here
        # pinned the worker on old code with nothing able to move it.
        print("Worker build id is unknown; treating the published build as newer.", flush=True)
        return True
    return remote_build != local_build


def _local_version_candidates() -> list[Path]:
    """Every file that can tell us which build this worker is running.

    worker_version.json ships inside the package, but the launcher records the
    build it just installed in .worker_state.json instead - so the file the
    updater writes and the file this check reads were not the same one. Read
    both, from the working directory and from the package root, since a worker
    started without the launcher's chdir would otherwise see neither. An unknown
    local build makes the update check below answer "no" forever, which is how a
    worker ends up stuck on old code with nothing in its log.
    """
    names = [LOCAL_VERSION_PATH.name, LOCAL_STATE_PATH.name]
    roots = [Path.cwd(), Path(__file__).resolve().parents[2]]
    candidates: list[Path] = []
    for name in names:
        for root in roots:
            path = root / name
            if path not in candidates:
                candidates.append(path)
    return candidates


def _local_build_id() -> str:
    for path in _local_version_candidates():
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        build_id = str(data.get("build_id") or data.get("sha256") or "").strip()
        if build_id:
            return build_id
    # Staying quiet here is what let a stalled worker keep running old code
    # unnoticed: no local build means the update check below always says no.
    print(
        "Worker build id is unknown (worker_version.json missing or unreadable); "
        "auto-update cannot run until it is restored.",
        flush=True,
    )
    return ""


def _default_worker_id() -> str:
    user = getpass.getuser()
    host = socket.gethostname()
    return f"{host}-{user}"


def _ensure_windows_autostart() -> None:
    if os.name != "nt":
        return
    if os.environ.get("LALADUB_WINDOWS_SERVICE") == "1":
        # Under a service the service is the autostart. Reinstalling the
        # scheduled task alongside it puts two supervisors on the same lock -
        # and this is the third place that did so, after the launcher and its
        # standalone installer, which is why the guard lives in here rather
        # than at each call site.
        return
    installer = Path.cwd() / "Install-Worker-Autostart.ps1"
    if not installer.is_file():
        return
    try:
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-WindowStyle",
                "Hidden",
                "-File",
                str(installer),
            ],
            cwd=Path.cwd(),
            check=False,
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:
        print(f"Worker autostart setup skipped: {type(exc).__name__}: {exc}", flush=True)


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._ ")
    return value[:180] or "item"


if __name__ == "__main__":
    main()
