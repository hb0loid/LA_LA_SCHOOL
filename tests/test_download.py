from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from laladub.download import _browser_cookie_config


class BrowserCookieConfigTests(unittest.TestCase):
    def test_explicit_firefox_profile_is_preserved(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LALADUB_YTDLP_BROWSER_COOKIES": "firefox",
                "LALADUB_YTDLP_BROWSER_PROFILE": r"C:\Users\Example\Firefox\profile",
            },
            clear=False,
        ):
            self.assertEqual(
                _browser_cookie_config(),
                ("firefox", str(Path(r"C:\Users\Example\Firefox\profile")), None, None),
            )

    def test_empty_browser_disables_cookie_extraction(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LALADUB_YTDLP_BROWSER_COOKIES": "",
                "LALADUB_YTDLP_BROWSER_PROFILE": r"C:\ignored",
            },
            clear=False,
        ):
            self.assertIsNone(_browser_cookie_config())
