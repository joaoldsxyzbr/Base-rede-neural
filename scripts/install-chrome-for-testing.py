"""Install the official Chrome for Testing Linux build into a temporary CI directory.

Only uses Python stdlib. Prints CHROME_BIN=<absolute_path> for GITHUB_ENV.
Not required by the Neural Core library or its consumers.
"""
import json
from pathlib import Path
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile

MANIFEST_URL = 'https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json'
MAX_BYTES = 300_000_000


def install(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(MANIFEST_URL, timeout=25) as response:
        manifest = json.load(response)
    versions = manifest['channels']['Stable']['downloads']['chrome']
    link = next(item['url'] for item in versions if item['platform'] == 'linux64')
    parsed = urllib.parse.urlparse(link)
    if parsed.scheme != 'https' or parsed.hostname != 'storage.googleapis.com':
        raise ValueError('Unexpected Chrome for Testing download host')

    with tempfile.TemporaryFile() as archive:
        with urllib.request.urlopen(link, timeout=60) as response:
            if int(response.headers.get('Content-Length', 0)) > MAX_BYTES:
                raise ValueError('Chrome download is too large')
            count = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > MAX_BYTES:
                    raise ValueError('Chrome download exceeded size limit')
                archive.write(chunk)
        archive.seek(0)
        with zipfile.ZipFile(archive) as zip_file:
            for member in zip_file.infolist():
                path = (destination / member.filename).resolve()
                if not path.is_relative_to(destination):
                    raise ValueError('Unsafe archive member')
            zip_file.extractall(destination)
    chrome = destination / 'chrome-linux64' / 'chrome'
    if not chrome.is_file():
        raise ValueError('Chrome for Testing executable is missing')
    for binary in ('chrome', 'chrome_crashpad_handler', 'chrome_sandbox'):
        target = destination / 'chrome-linux64' / binary
        if target.exists():
            target.chmod(target.stat().st_mode | 0o111)
    return chrome


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python scripts/install-chrome-for-testing.py DESTINATION')
    print(f'CHROME_BIN={install(sys.argv[1])}')
