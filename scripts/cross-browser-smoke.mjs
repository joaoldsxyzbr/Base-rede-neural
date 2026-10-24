/** Firefox/WebKit reais em Linux via Playwright oficial, somente no CI. */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { firefox, webkit } from 'playwright';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const routes = new Map([
  ['/tests/browser-smoke.html', ['tests/browser-smoke.html', 'text/html; charset=utf-8']],
  ['/tests/fixtures/parity.json', ['tests/fixtures/parity.json', 'application/json']],
  ['/src/index.js', ['src/index.js', 'text/javascript']],
  ['/src/model.js', ['src/model.js', 'text/javascript']],
  ['/src/network.js', ['src/network.js', 'text/javascript']]
]);

const server = createServer(async (request, response) => {
  const pathname = (request.url || '').split('?')[0];
  const route = routes.get(pathname);
  if (request.method !== 'GET' || !route) {
    response.writeHead(404).end('Not found');
    return;
  }
  try {
    response.writeHead(200, {
      'Content-Type': route[1],
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff'
    }).end(await readFile(join(root, route[0])));
  } catch {
    response.writeHead(500).end('Internal error');
  }
});

async function check(name, browserType, url) {
  const browser = await browserType.launch({ headless: true });
  try {
    const context = await browser.newContext();
    try {
      // Impede chamadas de rede externas: apenas nosso servidor loopback é autorizado.
      await context.route('**/*', route => {
        const target = new URL(route.request().url());
        if (target.hostname === '127.0.0.1' && target.protocol === 'http:') {
          return route.continue();
        }
        return route.abort();
      });
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 20_000 });
      await page.waitForFunction(() => {
        const status = document.querySelector('#result')?.dataset.status;
        return status === 'pass' || status === 'fail';
      }, { timeout: 20_000 });
      const state = await page.locator('#result').evaluate(el => ({
        status: el.dataset.status, text: el.textContent
      }));
      if (state.status !== 'pass' || errors.length) {
        throw new Error(`${name}: ${state.text}; pageerrors: ${errors.join(' | ')}`);
      }
      if (!state.text.includes('previsões')) throw new Error(`${name}: resultado inesperado`);
      process.stdout.write(`${name}: ${state.text}\n`);
    } finally {
      await context.close();
    }
  } finally {
    await browser.close();
  }
}

try {
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const url = `http://127.0.0.1:${server.address().port}/tests/browser-smoke.html`;
  for (const [name, type] of [['Firefox (Playwright build)', firefox],
                               ['WebKit (Linux Playwright build)', webkit]]) {
    await check(name, type, url);
  }
} finally {
  await new Promise(resolve => server.close(resolve));
}
