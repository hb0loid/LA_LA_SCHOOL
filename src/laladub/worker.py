from __future__ import annotations

import argparse
import getpass
import http.client
import json
import os
import re
import shutil
import socket
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .bot_config import load_bot_settings
from .job_runner import execute_job, result_manifest, save_job_snapshot


DEFAULT_CONFIG_PATH = Path("worker_config.json")
LOCAL_VERSION_PATH = Path("worker_version.json")
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

    client = CoordinatorClient(server, token)
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"LaLaDub worker started: id={worker_id} server={server} workdir={workdir}", flush=True)

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


class CoordinatorClient:
    def __init__(self, server: str, token: str) -> None:
        self.server = server.rstrip("/")
        self.token = token
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
            },
        )

    def upload_file(self, job_id: str, kind: str, path: Path) -> None:
        query = urllib.parse.urlencode({"filename": path.name})
        request_path = f"/api/v1/jobs/{urllib.parse.quote(job_id)}/result/{urllib.parse.quote(kind)}?{query}"
        self._upload_file(request_path, path)

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
    ) -> Any:
        data = None
        headers = self._auth_headers()
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        request = urllib.request.Request(self._url(path), data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status == 204 and none_on_204:
                    return None
                body = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Coordinator HTTP {exc.code}: {body}") from exc
        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    def _upload_file(self, path: str, file_path: Path) -> None:
        connection_cls = http.client.HTTPSConnection if self._parsed.scheme == "https" else http.client.HTTPConnection
        host = self._parsed.hostname or "127.0.0.1"
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
            response = connection.getresponse()
            body = response.read().decode("utf-8", errors="replace")
            if response.status >= 300:
                raise RuntimeError(f"Coordinator upload HTTP {response.status}: {body}")
        finally:
            connection.close()

    def _url(self, path: str) -> str:
        base_path = self._parsed.path.rstrip("/")
        return urllib.parse.urlunparse(
            (
                self._parsed.scheme,
                self._parsed.netloc,
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

    try:
        client.download_input(job_id, input_path)
        job = dict(remote_job)
        job["job_dir"] = str(job_dir)
        job["input_path"] = str(input_path)
        save_job_snapshot(job_dir, job, status="running")
        settings = load_bot_settings(require_token=False)
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
    return remote_build != local_build


def _local_build_id() -> str:
    try:
        data = json.loads(LOCAL_VERSION_PATH.read_text(encoding="utf-8-sig"))
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("build_id") or data.get("sha256") or "").strip()


def _default_worker_id() -> str:
    user = getpass.getuser()
    host = socket.gethostname()
    return f"{host}-{user}"


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._ ")
    return value[:180] or "item"


if __name__ == "__main__":
    main()
