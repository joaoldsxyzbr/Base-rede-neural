/** Real unpacked Manifest V3 smoke test. No npm dependencies or server required. */
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { access, copyFile, mkdtemp, mkdir, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const template = join(root, 'tests', 'extension-mv3');
const artifact = await mkdtemp(join(tmpdir(), 'neural-core-mv3-'));
const extension = join(artifact, 'extension');
const profile = join(artifact, 'profile');

async function resolveBrowser() {
  for (const candidate of [process.env.CHROME_BIN, '/usr/bin/chromium', '/usr/bin/google-chrome']) {
    if (!candidate) continue;
    try { await access(candidate); return candidate; } catch { /* Try the next binary. */ }
  }
  throw new Error('Chrome for Testing/Chromium missing; specify CHROME_BIN');
}

function extensionId(publicKey) {
  const der = Buffer.from(publicKey, 'base64');
  if (der.length < 100) throw new Error('Invalid public development key');
  const hash = createHash('sha256').update(der).digest('hex').slice(0, 32);
  return [...hash].map(nibble => 'abcdefghijklmnop'[parseInt(nibble, 16)]).join('');
}

function invokeBrowser(binary, url) {
  return new Promise((resolve, reject) => {
    const args = [
      '--headless=new', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
      '--disable-background-networking', '--disable-sync', '--no-first-run',
      '--no-default-browser-check', '--disable-default-apps',
      `--user-data-dir=${profile}`,
      `--disable-extensions-except=${extension}`, `--load-extension=${extension}`,
      '--virtual-time-budget=15000', '--dump-dom', url
    ];
    const child = spawn(binary, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    let timedOut = false;
    const deadline = setTimeout(() => { timedOut = true; child.kill('SIGKILL'); }, 45000);
    child.stdout.on('data', chunk => { stdout = (stdout + chunk).slice(0, 1_000_000); });
    child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-3000); });
    child.once('error', error => { clearTimeout(deadline); reject(error); });
    child.once('close', code => {
      clearTimeout(deadline);
      if (timedOut) return reject(new Error('Extension Chromium smoke timed out; browser result unverified'));
      if (code !== 0) return reject(new Error(`Chrome exited ${code}: ${stderr.slice(-1000)}`));
      resolve(stdout);
    });
  });
}

try {
  await mkdir(join(extension, 'src'), { recursive: true });
  await mkdir(profile, { recursive: true });
  for (const name of ['manifest.json', 'popup.html', 'popup.js', 'worker.js']) {
    await copyFile(join(template, name), join(extension, name));
  }
  for (const name of ['index.js', 'model.js', 'network.js']) {
    await copyFile(join(root, 'src', name), join(extension, 'src', name));
  }
  await copyFile(join(root, 'tests', 'fixtures', 'parity.json'), join(extension, 'fixture.json'));

  const manifest = JSON.parse(await readFile(join(extension, 'manifest.json'), 'utf8'));
  if (manifest.manifest_version !== 3 ||
      manifest.background?.type !== 'module' ||
      manifest.permissions || manifest.host_permissions) {
    throw new Error('Manifest is not minimal MV3');
  }
  const url = `chrome-extension://${extensionId(manifest.key)}/popup.html`;
  const dom = await invokeBrowser(await resolveBrowser(), url);
  const output = dom.match(/<output\b([^>]*)>([^<]*)<\/output>/i);
  if (!output || !/\bdata-status="pass"/.test(output[1]) ||
      !output[2].includes('MV3 popup (19 cases) and service worker')) {
    throw new Error(`MV3 extension failed: ${output?.[2] || 'No extension output (is the unpacked extension loaded?)'}`);
  }
  process.stdout.write(`${output[2]}\n`);
} finally {
  await rm(artifact, { recursive: true, force: true });
}
