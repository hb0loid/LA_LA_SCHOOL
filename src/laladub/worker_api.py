from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def start_worker_api(application: Any, *, host: str, port: int, token: str) -> ThreadingHTTPServer:
    if not token:
        raise RuntimeError("LALADUB_WORKER_API_TOKEN is required when remote workers are enabled.")
    loop = asyncio.get_running_loop()
    context = _ApplicationContext(application)
    server = _WorkerHTTPServer((host, port), _WorkerRequestHandler, application, context, loop, token)
    thread = threading.Thread(target=server.serve_forever, name="laladub-worker-api", daemon=True)
    thread.start()
    print(f"Worker API listening on http://{host}:{port}", flush=True)
    return server


class _ApplicationContext:
    def __init__(self, application: Any) -> None:
        self.application = application
        self.bot = application.bot


class _WorkerHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        application: Any,
        context: _ApplicationContext,
        loop: asyncio.AbstractEventLoop,
        token: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.application = application
        self.context = context
        self.loop = loop
        self.token = token


class _WorkerRequestHandler(BaseHTTPRequestHandler):
    server: _WorkerHTTPServer

    def do_GET(self) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/health":
                self._send_json({"ok": True})
                return
            if not self._authorized():
                self._send_error(401, "unauthorized")
                return
            if parsed.path == "/api/v1/worker/manifest":
                self._send_json(self._worker_manifest())
                return
            if parsed.path == "/api/v1/worker/package":
                package_path = self._worker_package_path()
                if package_path is None:
                    self._send_error(404, "worker package not found")
                    return
                self._send_file(package_path)
                return
            if parsed.path == "/api/v1/jobs/lease":
                query = urllib.parse.parse_qs(parsed.query)
                worker_id = (query.get("worker_id") or ["worker"])[0]
                result = self._run_coro(self._scheduler().lease_remote(self.server.context, worker_id))
                if result is None:
                    self.send_response(204)
                    self.end_headers()
                else:
                    self._send_json(result)
                return
            job_id, suffix = self._match_job_path(parsed.path)
            if job_id and suffix == "input":
                input_path = self._run_coro(self._scheduler().remote_input_path(job_id))
                if input_path is None:
                    self._send_error(404, "job input not found")
                    return
                self._send_file(input_path)
                return
            self._send_error(404, "not found")
        except Exception as exc:
            self._send_error(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            if not self._authorized():
                self._send_error(401, "unauthorized")
                return
            job_id, suffix = self._match_job_path(parsed.path)
            if not job_id:
                self._send_error(404, "not found")
                return
            payload = self._read_json()
            if suffix == "progress":
                self._run_coro(self._scheduler().remote_progress(job_id, payload))
                self._send_json({"ok": True})
                return
            if suffix == "complete":
                self._run_coro(self._scheduler().complete_remote(self.server.context, job_id, payload))
                self._send_json({"ok": True})
                return
            if suffix == "fail":
                self._run_coro(self._scheduler().fail_remote(self.server.context, job_id, payload))
                self._send_json({"ok": True})
                return
            self._send_error(404, "not found")
        except Exception as exc:
            self._send_error(500, f"{type(exc).__name__}: {exc}")

    def do_PUT(self) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            if not self._authorized():
                self._send_error(401, "unauthorized")
                return
            job_id, suffix = self._match_job_path(parsed.path)
            if not job_id or not suffix.startswith("result/"):
                self._send_error(404, "not found")
                return
            kind = suffix.split("/", 1)[1]
            query = urllib.parse.parse_qs(parsed.query)
            filename = (query.get("filename") or ["upload.bin"])[0]
            output_path = self._run_coro(self._scheduler().remote_upload_path(job_id, kind, filename))
            if output_path is None:
                self._send_error(404, "job not found")
                return
            self._receive_file(output_path)
            self._send_json({"ok": True, "filename": output_path.name})
        except Exception as exc:
            self._send_error(500, f"{type(exc).__name__}: {exc}")

    def log_message(self, format: str, *args: Any) -> None:
        print(f"Worker API: {self.address_string()} - {format % args}", flush=True)

    def _scheduler(self) -> Any:
        return self.server.application.bot_data["job_scheduler"]

    def _settings(self) -> Any:
        return self.server.application.bot_data["settings"]

    def _worker_package_path(self) -> Path | None:
        path = Path(self._settings().worker_package_path)
        if path.exists() and path.is_file():
            return path
        return None

    def _worker_manifest(self) -> dict[str, Any]:
        settings = self._settings()
        package_path = self._worker_package_path()
        manifest_path = Path(settings.worker_package_manifest_path)
        manifest = _read_json_file(manifest_path)
        data: dict[str, Any] = {
            "available": package_path is not None,
            "version": str(manifest.get("version") or settings.worker_version),
            "build_id": str(manifest.get("build_id") or ""),
            "package": "/api/v1/worker/package",
        }
        if package_path is None:
            return data
        size = package_path.stat().st_size
        sha256 = str(manifest.get("sha256") or "") or _sha256_file(package_path)
        data.update(
            {
                "filename": package_path.name,
                "size": size,
                "sha256": sha256,
                "build_id": data["build_id"] or sha256[:16],
            }
        )
        return data

    def _run_coro(self, coro: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self.server.loop)
        return future.result(timeout=30)

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        return header == f"Bearer {self.server.token}"

    def _match_job_path(self, path: str) -> tuple[str | None, str]:
        prefix = "/api/v1/jobs/"
        if not path.startswith(prefix):
            return None, ""
        remainder = path[len(prefix) :].strip("/")
        if not remainder:
            return None, ""
        parts = remainder.split("/", 1)
        job_id = urllib.parse.unquote(parts[0])
        suffix = urllib.parse.unquote(parts[1]) if len(parts) > 1 else ""
        return job_id, suffix

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        body = self.rfile.read(length)
        data = json.loads(body.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON object expected")
        return data

    def _receive_file(self, output_path: Path) -> None:
        length = int(self.headers.get("Content-Length") or "0")
        if length < 0:
            raise ValueError("Bad Content-Length")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        remaining = length
        with output_path.open("wb") as output:
            while remaining > 0:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                output.write(chunk)
                remaining -= len(chunk)
        if remaining:
            raise IOError(f"Upload ended early, {remaining} bytes missing")

    def _send_file(self, path: Path) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        with path.open("rb") as input_file:
            shutil.copyfileobj(input_file, self.wfile)

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json({"ok": False, "error": message}, status=status)


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
