"""Offline tests for CI Chrome for Testing download boundaries."""
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

_path = Path(__file__).resolve().parents[1] / "scripts" / "install-chrome-for-testing.py"
_spec = importlib.util.spec_from_file_location("pinned_chrome_installer", _path)
_installer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_installer)
CHROME_VERSION = _installer.CHROME_VERSION
CHROME_URL = _installer.CHROME_URL
check_download_url = _installer.check_download_url
safe_members = _installer.safe_members


class InstallerTests(unittest.TestCase):
    def test_url_restricts_host_platform_and_version(self):
        check_download_url(CHROME_URL)
        for url in (
            "http://storage.googleapis.com/chrome-for-testing-public/" + CHROME_VERSION + "/linux64/chrome-linux64.zip",
            "https://evil.example.com/chrome.zip",
            "https://storage.googleapis.com/chrome-for-testing-public/0.0.0.0/linux64/chrome-linux64.zip",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                check_download_url(url)

    def test_zip_rejects_parent_escape(self):
        blob = io.BytesIO()
        with zipfile.ZipFile(blob, "w") as archive:
            archive.writestr("../../escape.txt", "test")
        blob.seek(0)
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(blob) as archive:
            with self.assertRaises(ValueError):
                safe_members(archive, Path(tmp).resolve())


if __name__ == "__main__":
    unittest.main()
