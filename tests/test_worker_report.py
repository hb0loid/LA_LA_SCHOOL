from __future__ import annotations

import tempfile
import unittest
import unittest.mock
from pathlib import Path

from laladub.worker import REPORT_TAIL_BYTES, _log_tail, _worker_log_dir


class WorkerReportTests(unittest.TestCase):
    """The laptop answers nothing from outside, so when it dies the reason is
    written on a machine the other side cannot read. The worker carries it over
    on the way back up."""

    def test_a_short_log_is_sent_whole(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "worker.log"
            path.write_bytes(b"first line\nsecond line\n")
            self.assertEqual(_log_tail(path).splitlines(), ["first line", "second line"])

    def test_a_long_log_is_cut_to_the_tail_at_a_line_break(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "worker.log"
            path.write_text("x" * REPORT_TAIL_BYTES + "\nthe part that matters\n", encoding="utf-8")
            tail = _log_tail(path)
            self.assertEqual(tail, "the part that matters")

    def test_a_missing_log_is_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            self.assertEqual(_log_tail(Path(tempdir) / "nope.log"), "")

    def test_the_log_dir_is_found_two_levels_above_the_workdir(self) -> None:
        """The launcher runs the worker from the package root and passes
        runs/worker as its workdir."""
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "logs").mkdir()
            (root / "runs" / "worker").mkdir(parents=True)
            self.assertEqual(_worker_log_dir(root / "runs" / "worker"), root / "logs")



class ContactStampTests(unittest.TestCase):
    """The supervisor kills a worker whose stamp goes stale, so the stamp has to
    mean "we reached the coordinator" - not "this process is alive".

    The worker addresses the main PC by name, and a name lookup has no timeout
    and does not hold the interpreter. A lookup that never returned left every
    thread running happily while the worker said nothing to anyone for thirty
    hours, and a watchdog keyed on liveness saw a healthy process.
    """

    def test_a_successful_request_stamps_the_file(self) -> None:
        from laladub.worker import CoordinatorClient

        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "worker_heartbeat.txt"
            client = CoordinatorClient("http://127.0.0.1:8765", "token", "worker-pc")
            client.contact_path = path
            client.note_contact()
            self.assertTrue(path.exists())
            self.assertGreater(float(path.read_text(encoding="utf-8")), 0.0)

    def test_stamping_is_throttled(self) -> None:
        from laladub.worker import CoordinatorClient

        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "worker_heartbeat.txt"
            client = CoordinatorClient("http://127.0.0.1:8765", "token", "worker-pc")
            client.contact_path = path
            client.note_contact()
            first = path.read_text(encoding="utf-8")
            client.note_contact()
            self.assertEqual(path.read_text(encoding="utf-8"), first)

    def test_no_path_is_harmless(self) -> None:
        from laladub.worker import CoordinatorClient

        client = CoordinatorClient("http://127.0.0.1:8765", "token", "worker-pc")
        client.note_contact()


class LogEncodingTests(unittest.TestCase):
    """Windows PowerShell 5.1 redirects native output as UTF-16, so the first
    report that arrived read as text with a space between every letter."""

    def test_a_utf16_log_is_read_as_text(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "worker.log"
            path.write_bytes("﻿worker started\nDetected language: ru\n".encode("utf-16-le"))
            self.assertEqual(
                _log_tail(path).splitlines(), ["worker started", "Detected language: ru"]
            )

    def test_a_utf8_log_still_reads_as_before(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "worker.log"
            path.write_bytes("worker started\nDetected language: ru\n".encode("utf-8"))
            self.assertEqual(
                _log_tail(path).splitlines(), ["worker started", "Detected language: ru"]
            )

    def test_a_long_utf16_tail_does_not_land_mid_character(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "worker.log"
            body = "﻿" + ("filler line\n" * 2000) + "the part that matters\n"
            path.write_bytes(body.encode("utf-16-le"))
            self.assertTrue(_log_tail(path).endswith("the part that matters"))



class HostResolutionTests(unittest.TestCase):
    """A name lookup has no timeout and is not covered by the socket timeout, so
    doing one per request is what let a single wedged lookup silence the worker
    for thirty hours. It happens once now, and only again after a failure."""

    def test_an_address_is_used_as_is(self) -> None:
        from laladub.worker import CoordinatorClient

        client = CoordinatorClient("http://192.168.1.180:8765", "token", "worker-pc")
        with unittest.mock.patch("socket.getaddrinfo", side_effect=AssertionError("looked up")):
            self.assertEqual(client._connect_host(), "192.168.1.180")

    def test_a_name_is_looked_up_once_and_reused(self) -> None:
        from laladub.worker import CoordinatorClient

        client = CoordinatorClient("http://HBoloid:8765", "token", "worker-pc")
        lookups: list[str] = []

        def fake(host, port, **kwargs):
            lookups.append(host)
            return [(0, 0, 0, "", ("192.168.1.180", port))]

        with unittest.mock.patch("socket.getaddrinfo", side_effect=fake):
            self.assertEqual(client._connect_host(), "192.168.1.180")
            self.assertEqual(client._connect_host(), "192.168.1.180")
        self.assertEqual(len(lookups), 1)

    def test_a_failure_forces_a_fresh_lookup(self) -> None:
        from laladub.worker import CoordinatorClient

        client = CoordinatorClient("http://HBoloid:8765", "token", "worker-pc")
        addresses = ["192.168.1.180", "192.168.1.181"]

        def fake(host, port, **kwargs):
            return [(0, 0, 0, "", (addresses.pop(0), port))]

        with unittest.mock.patch("socket.getaddrinfo", side_effect=fake):
            self.assertEqual(client._connect_host(), "192.168.1.180")
            client.forget_host()
            self.assertEqual(client._connect_host(), "192.168.1.181")

    def test_a_lookup_that_fails_keeps_the_last_good_address(self) -> None:
        from laladub.worker import CoordinatorClient

        client = CoordinatorClient("http://HBoloid:8765", "token", "worker-pc")
        with unittest.mock.patch(
            "socket.getaddrinfo", return_value=[(0, 0, 0, "", ("192.168.1.180", 8765))]
        ):
            client._connect_host()
        client.forget_host()
        with unittest.mock.patch("socket.getaddrinfo", side_effect=OSError("no dns")):
            self.assertEqual(client._connect_host(), "192.168.1.180")



class ServiceModeTests(unittest.TestCase):
    """Under a Windows service the service is the autostart. Reinstalling the
    scheduled task alongside it puts two supervisors on one lock, and this was
    the third place doing so - after the launcher and its standalone installer -
    which is why the guard sits inside the function, not at the call sites."""

    def test_the_scheduled_task_is_not_installed_in_service_mode(self) -> None:
        import subprocess as sp

        from laladub.worker import _ensure_windows_autostart

        with unittest.mock.patch.dict("os.environ", {"LALADUB_WINDOWS_SERVICE": "1"}):
            with unittest.mock.patch.object(sp, "run", side_effect=AssertionError("ran")):
                _ensure_windows_autostart()

    def test_it_is_installed_without_the_variable(self) -> None:
        import os
        import subprocess as sp

        from laladub.worker import _ensure_windows_autostart

        if os.name != "nt":
            self.skipTest("Windows only")
        calls: list[object] = []
        environment = dict(os.environ)
        environment.pop("LALADUB_WINDOWS_SERVICE", None)
        with unittest.mock.patch.dict("os.environ", environment, clear=True):
            with unittest.mock.patch.object(Path, "is_file", return_value=True):
                with unittest.mock.patch.object(sp, "run", side_effect=lambda *a, **k: calls.append(a)):
                    _ensure_windows_autostart()
        self.assertEqual(len(calls), 1)

if __name__ == "__main__":
    unittest.main()
