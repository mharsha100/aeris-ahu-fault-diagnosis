"""Offline control tests, not evidence of publisher access or fresh setup."""

import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location("clean_check", Path(__file__).with_name("clean_data_check.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("fixture.txt", "synthetic offline fixture")
        self.content = stream.getvalue()
        self.snapshot = {"name": "fixture.zip", "url": "https://example.invalid/fixture.zip",
                         "bytes": len(self.content), "sha256": hashlib.sha256(self.content).hexdigest()}

    def test_verified_fresh_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "new"
            with patch.object(module.urllib.request, "urlopen", return_value=io.BytesIO(self.content)):
                final, report = module.download_verified(self.snapshot, folder)
            self.assertEqual(final.read_bytes(), self.content)
            self.assertEqual(report["action"], "downloaded_uncached_and_verified")
            self.assertEqual(report["crc_check"], "passed")

    def test_existing_directory_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(module.urllib.request, "urlopen") as opener:
                with self.assertRaises(FileExistsError):
                    module.download_verified(self.snapshot, Path(tmp))
                opener.assert_not_called()

    def test_bad_hash_does_not_publish_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "new"
            snapshot = dict(self.snapshot, sha256="0" * 64)
            with patch.object(module.urllib.request, "urlopen", return_value=io.BytesIO(self.content)):
                with self.assertRaisesRegex(ValueError, "snapshot mismatch"):
                    module.download_verified(snapshot, folder)
            self.assertFalse((folder / "fixture.zip").exists())

    def test_oversized_response_is_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "new"
            with patch.object(module.urllib.request, "urlopen", return_value=io.BytesIO(self.content + b"extra")):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    module.download_verified(self.snapshot, folder)
            self.assertLessEqual((folder / "fixture.zip.partial").stat().st_size, self.snapshot["bytes"])

    def test_original_audit_hashes_detect_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "milestone3/evidence"
            folder.mkdir(parents=True)
            record = folder / "run.json"
            record.write_text('{"version":"m3-v1"}')
            before = module.original_audit_hashes(root)
            record.write_text('{"version":"m3-v1", "changed":true}')
            self.assertNotEqual(before, module.original_audit_hashes(root))


if __name__ == "__main__":
    unittest.main()
