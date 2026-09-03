from __future__ import annotations

import http.client
import unittest
from pathlib import Path
from unittest.mock import patch

from laladub.worker import UPLOAD_ATTEMPTS, CoordinatorClient


class UploadRetryTests(unittest.TestCase):
    """A dropped connection cost a whole job's work.

    The laptop finished preprocessing, the link died mid-upload with
    ConnectionAborted, and everything it had computed was thrown away and redone
    on the main PC. The upload is a PUT to a fixed path, so retrying is safe.
    """

    def setUp(self) -> None:
        self.client = CoordinatorClient("http://127.0.0.1:8765", "token", "worker-pc")

    def test_a_dropped_connection_is_retried(self) -> None:
        attempts: list[int] = []

        def flaky(self: object, path: str, file_path: Path) -> None:
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionAbortedError(10053, "An established connection was aborted")

        with patch.object(CoordinatorClient, "_upload_file", flaky), patch("time.sleep"):
            self.client.upload_file("job", "documents", Path("bundle.zip"))
        self.assertEqual(len(attempts), 3)

    def test_it_gives_up_eventually(self) -> None:
        def always_broken(self: object, path: str, file_path: Path) -> None:
            raise http.client.HTTPException("connection reset")

        with patch.object(CoordinatorClient, "_upload_file", always_broken), patch("time.sleep"):
            with self.assertRaises(http.client.HTTPException):
                self.client.upload_file("job", "documents", Path("bundle.zip"))

    def test_a_real_error_is_not_retried(self) -> None:
        calls: list[int] = []

        def refused(self: object, path: str, file_path: Path) -> None:
            calls.append(1)
            raise RuntimeError("Coordinator upload HTTP 404: unknown job")

        with patch.object(CoordinatorClient, "_upload_file", refused), patch("time.sleep"):
            with self.assertRaises(RuntimeError):
                self.client.upload_file("job", "documents", Path("bundle.zip"))
        self.assertEqual(len(calls), 1)
        self.assertGreater(UPLOAD_ATTEMPTS, 1)


if __name__ == "__main__":
    unittest.main()
