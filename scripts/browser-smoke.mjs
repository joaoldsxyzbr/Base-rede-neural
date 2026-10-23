/** Smoke test real no Chrome/Chromium headless, sem dependências npm. */
import { createServer } from 'node:http';
import { readFile, access } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const routes = new Map([
  ['/tests/browser-smoke.html', ['tests/browser-smoke.html', 'text/html; charset=utf-8']],
  ['/tests/fixtures/parity.json', ['tests/fixtures/parity.json', 'application/json']],
  ['/src/index.js', ['src/index.js', 'text/javascript']],
  ['/src/model.js', ['src/model.js', 'text/javascript']],
  ['/src/network.js', ['src/network.js', 'text/javascript']]
]);

async function findChrome() {
  for (const executable of [process.env.CHROME_BIN, '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium']) {
    if (!executable) continue;
    try { await access(executable); return executable; } catch { /* próximo candidato */ }
  }
  throw new Error('Chrome/Chromium ausente; informe CHROME_BIN para testar no navegador');
}

function runBrowser(executable, url) {
  return new Promise((resolve, reject) => {
    const child = spawn(executable, [
      '--headless', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
      '--disable-background-networking', '--no-first-run', '--no-default-browser-check',
      '--virtual-time-budget=10000', '--dump-dom', url
    ], { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    const limit = 1_000_000;
    const timer = setTimeout(() => child.kill('SIGKILL'), 35000);
    child.stdout.on('data', chunk => { stdout = (stdout + chunk).slice(0, limit); });
    child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-limit); });
    child.on('error', error => { clearTimeout(timer); reject(error); });
    child.on('close', code => {
      clearTimeout(timer);
      if (code !== 0) return reject(new Error(`Chrome encerrou com código ${code}: ${stderr.slice(-1500)}`));
      resolve(stdout);
    });
  });
}

const server = createServer(async (request, response) => {
  const route = routes.get((request.url || '').split('?')[0]);
  if (request.method !== 'GET' || !route) {
    response.writeHead(404).end('Not found');
    return;
  }
  try {
    const content = await readFile(join(root, route[0]));
    response.writeHead(200, {
      'Content-Type': route[1],
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff'
    }).end(content);
  } catch {
    response.writeHead(500).end('Internal error');
  }
});

try {
  const executable = await findChrome();
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const port = server.address().port;
  const dom = await runBrowser(executable, `http://127.0.0.1:${port}/tests/browser-smoke.html`);
  const result = dom.match(/<output id="result"[^>]*>([^<]*)<\/output>/);
  if (!result || !dom.includes('data-status="pass"')) {
    throw new Error(`Teste no navegador falhou: ${result?.[1] || 'sem resultado no DOM'}`);
  }
  process.stdout.write(`${result[1]}\n`);
} finally {
  if (server.listening) server.close();
}
