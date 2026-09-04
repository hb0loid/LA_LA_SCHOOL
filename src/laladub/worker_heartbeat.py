"""Independent network heartbeat for a busy LaLaDub worker lease."""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request


def main() -> None:
    server = os.environ["LALADUB_HEARTBEAT_SERVER"].rstrip("/")
    token = os.environ["LALADUB_HEARTBEAT_TOKEN"]
    worker_id = os.environ["LALADUB_HEARTBEAT_WORKER_ID"]
    job_id = os.environ["LALADUB_HEARTBEAT_JOB_ID"]
    url = f"{server}/api/v1/jobs/{urllib.parse.quote(job_id)}/progress"
    body = json.dumps({"heartbeat_only": True, "worker_id": worker_id}).encode("utf-8")

    while True:
        try:
            request = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=5):
                pass
        except Exception:
            # A short network interruption is expected; the next pulse retries.
            pass
        time.sleep(10.0)


if __name__ == "__main__":
    main()
