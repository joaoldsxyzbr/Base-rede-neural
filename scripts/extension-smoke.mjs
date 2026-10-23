/** Real unpacked Manifest V3 smoke test. No npm dependencies or server required. */
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { setTimeout as sleep } from 'node:timers/promises';
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

async function invokeBrowser(binary, url) {
  const args = [
    '--headless=new', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
    '--disable-background-networking', '--disable-sync', '--no-first-run',
    '--no-default-browser-check', '--disable-default-apps',
    '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=0',
    `--user-data-dir=${profile}`,
    `--disable-extensions-except=${extension}`, `--load-extension=${extension}`,
    'about:blank'
  ];
  const chrome = spawn(binary, args, { stdio: ['ignore', 'ignore', 'pipe'] });
  let stderr = '';
  chrome.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-2400); });
  let connection;
  try {
    // CDP advances in actual time. --dump-dom virtual time raced MV3 worker startup.
    let port;
    const deadline = Date.now() + 15000;
    while (Date.now() < deadline) {
      if (chrome.exitCode !== null) throw new Error(`Chrome exited early: ${stderr}`);
      try {
        const lines = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n');
        port = Number(lines[0]);
        if (Number.isInteger(port) && port > 0) break;
      } catch { /* Browser is starting. */ }
      await sleep(150);
    }
    if (!port) throw new Error(`Chrome debugging endpoint not available: ${stderr}`);
    const metadata = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
    connection = new WebSocket(metadata.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('CDP connection timed out')), 10000);
      connection.addEventListener('open', () => { clearTimeout(timeout); resolve(); }, { once: true });
      connection.addEventListener('error', event => { clearTimeout(timeout); reject(new Error(String(event.message || 'CDP websocket failed'))); }, { once: true });
    });
    const pending = new Map();
    let sequence = 0;
    connection.addEventListener('message', event => {
      const message = JSON.parse(event.data);
      const current = pending.get(message.id);
      if (!current) return;
      pending.delete(message.id);
      clearTimeout(current.timeout);
      if (message.error) current.reject(new Error(message.error.message));
      else current.resolve(message.result);
    });
    const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
      const id = ++sequence;
      const timeout = setTimeout(() => { pending.delete(id); reject(new Error(`CDP ${method} timed out`)); }, 10000);
      pending.set(id, { resolve, reject, timeout });
      connection.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    });
    const { targetId } = await send('Target.createTarget', { url });
    const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
    await send('Runtime.enable', {}, sessionId);
    const evaluate = "(() => { const el = document.getElementById('result'); return {status: el?.dataset.status || 'pending', text: el?.textContent || ''}; })()";
    const end = Date.now() + 30000;
    while (Date.now() < end) {
      const result = await send('Runtime.evaluate', { expression: evaluate, returnByValue: true }, sessionId);
      const state = result?.result?.value;
      if (state?.status === 'pass') return state.text;
      if (state?.status === 'fail') throw new Error(`MV3 extension failed: ${state.text}`);
      await sleep(250);
    }
    throw new Error('MV3 extension test timed out waiting for popup and service worker');
  } finally {
    connection?.close();
    if (chrome.exitCode === null && chrome.signalCode === null) {
      const exited = once(chrome, 'exit');
      chrome.kill('SIGKILL');
      await Promise.race([exited, sleep(3000)]);
    }
  }
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
  const result = await invokeBrowser(await resolveBrowser(), url);
  if (!result.includes('MV3 popup (19 cases) and service worker')) {
    throw new Error(`MV3 extension result was unexpected: ${result}`);
  }
  process.stdout.write(`${result}\n`);
} finally {
  await rm(artifact, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
}
