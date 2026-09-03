from __future__ import annotations

import os
import unittest


class ArgosChunkTypeTests(unittest.TestCase):
    def test_importing_translation_selects_minisbd(self) -> None:
        # Argos reads ARGOS_CHUNK_TYPE once, at import. The default splitter
        # needs a Stanza model per language and there is none for Malay or
        # Azerbaijani, so reading either raised "No processors to load for
        # language ms" and killed the job whenever the online translator was
        # rate-limited. MiniSBD needs no per-language model.
        import laladub.translation  # noqa: F401

        self.assertEqual(os.environ.get("ARGOS_CHUNK_TYPE"), "MINISBD")

    def test_an_explicit_setting_is_respected(self) -> None:
        # setdefault, not an override: anyone debugging a chunking difference
        # can still pin it from the environment.
        import importlib

        previous = os.environ.get("ARGOS_CHUNK_TYPE")
        os.environ["ARGOS_CHUNK_TYPE"] = "STANZA"
        try:
            import laladub.translation

            importlib.reload(laladub.translation)
            self.assertEqual(os.environ.get("ARGOS_CHUNK_TYPE"), "STANZA")
        finally:
            if previous is None:
                os.environ.pop("ARGOS_CHUNK_TYPE", None)
            else:
                os.environ["ARGOS_CHUNK_TYPE"] = previous
            importlib.reload(laladub.translation)


if __name__ == "__main__":
    unittest.main()
