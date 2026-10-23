"""Install a pinned Chrome for Testing build into an ephemeral CI directory.

No third-party Python dependencies; downloads are for CI testing only.
"""
import hashlib
from pathlib import Path
import stat
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile

CHROME_VERSION = "156.0.8078.47"
CHROME_SHA256 = "e6c94490f0b6b337225aeb2d2a5034858516b7017356980ba94f5176d8774a50"  # Verified Chrome for Testing v156 linux64 artifact.
CHROME_URL = (
    "https://storage.googleapis.com/chrome-for-testing-public/"
    f"{CHROME_VERSION}/linux64/chrome-linux64.zip"
)
MAX_BYTES = 300_000_000


def check_download_url(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "storage.googleapis.com":
        raise ValueError("Origem do Chrome for Testing não autorizada")
    if not parsed.path.startswith(f"/chrome-for-testing-public/{CHROME_VERSION}/linux64/"):
        raise ValueError("Versão ou plataforma inesperada")


def safe_members(archive, destination):
    for item in archive.infolist():
        resolved = (destination / item.filename).resolve()
        if not resolved.is_relative_to(destination):
            raise ValueError("Arquivo zip contém caminho inseguro")
        file_type = (item.external_attr >> 16) & 0o170000
        if file_type == stat.S_IFLNK:
            raise ValueError("Arquivo zip contém link simbólico")


def install(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    check_download_url(CHROME_URL)
    digest = hashlib.sha256()
    with tempfile.TemporaryFile() as archive:
        with urllib.request.urlopen(CHROME_URL, timeout=60) as response:
            check_download_url(response.url)
            if int(response.headers.get("Content-Length", 0)) > MAX_BYTES:
                raise ValueError("Download do Chrome excede limite")
            count = 0
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > MAX_BYTES:
                    raise ValueError("Download do Chrome excede limite")
                archive.write(chunk)
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        print(f"Chrome for Testing {CHROME_VERSION} SHA256: {actual_sha256}", file=sys.stderr)
        if actual_sha256 != CHROME_SHA256:
            raise ValueError("Hash SHA256 do Chrome diverge do esperado")
        archive.seek(0)
        with zipfile.ZipFile(archive) as package:
            safe_members(package, destination)
            package.extractall(destination)
    chrome = destination / "chrome-linux64" / "chrome"
    if not chrome.is_file():
        raise ValueError("Executável Chrome for Testing ausente")
    for binary in ("chrome", "chrome_crashpad_handler", "chrome_sandbox"):
        target = chrome.parent / binary
        if target.exists():
            target.chmod(target.stat().st_mode | 0o111)
    return chrome


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Use: python scripts/install-chrome-for-testing.py DESTINATION")
    print(f"CHROME_BIN={install(sys.argv[1])}")
