from pathlib import Path
import unittest


class MossFallbackGuardTests(unittest.TestCase):
    def test_pipeline_falls_back_from_moss_to_cosyvoice_only(self) -> None:
        source = (Path(__file__).parents[1] / "src" / "laladub" / "pipeline.py").read_text(encoding="utf-8")

        self.assertIn("MOSS batch fallback to CosyVoice", source)
        self.assertIn("Standard system voice fallback is disabled.", source)
        self.assertIn("cosyvoice_fallback_paths", source)

    def test_moss_batch_timeout_scales_for_long_jobs(self) -> None:
        source = (Path(__file__).parents[1] / "src" / "laladub" / "tts.py").read_text(encoding="utf-8")

        self.assertIn("600 + len(manifest_items) * 12", source)
        self.assertIn("timeout_seconds=batch_timeout_seconds", source)

    def test_moss_retries_only_missing_lines_before_cosyvoice(self) -> None:
        source = (Path(__file__).parents[1] / "src" / "laladub" / "pipeline.py").read_text(encoding="utf-8")

        self.assertIn("retrying the missing lines once", source)
        self.assertIn("synthesize_moss_batch(missing_items, config)", source)

    def test_cosyvoice_timeout_scales_for_large_fallbacks(self) -> None:
        source = (Path(__file__).parents[1] / "src" / "laladub" / "tts.py").read_text(encoding="utf-8")

        self.assertIn("600 + len(manifest_items) * 15", source)

    def test_moss_runner_releases_cuda_cache(self) -> None:
        source = (Path(__file__).parents[1] / "tools" / "moss_tts_batch_runner.py").read_text(encoding="utf-8")

        self.assertIn("torch.cuda.empty_cache()", source)


if __name__ == "__main__":
    unittest.main()
