/** Testa o Neural Core no workerd local via Wrangler oficial, sem conta/deploy. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createServer } from 'node:net';
import { setTimeout as sleep } from 'node:timers/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const cli = join(root, 'node_modules', 'wrangler', 'bin', 'wrangler.js');

const port = await new Promise((resolve, reject) => {
  const server = createServer();
  server.once('error', reject);
  server.listen(0, '127.0.0.1', () => {
    const selected = server.address().port;
    server.close(() => resolve(selected));
  });
});
const child = spawn(process.execPath, [
  cli, 'dev', 'tests/worker-smoke.js', '--local',
  '--ip', '127.0.0.1', '--port', String(port),
  '--compatibility-date', '2026-10-24'
], {
  cwd: root,
  env: {
    ...process.env, WRANGLER_SEND_METRICS: 'false',
    CLOUDFLARE_SEND_METRICS: 'false', CI: 'true'
  },
  stdio: ['ignore', 'pipe', 'pipe']
});
let logs = '';
for (const stream of [child.stdout, child.stderr]) {
  stream.on('data', chunk => { logs = (logs + chunk).slice(-4000); });
}
try {
  const deadline = Date.now() + 50_000;
  let lastError = '';
  let passed = false;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(`Wrangler terminou antes do teste (código ${child.exitCode}): ${logs}`);
    }
    try {
      const response = await fetch(`http://127.0.0.1:${port}/`, {
        signal: AbortSignal.timeout(2500)
      });
      const result = await response.json();
      assert.equal(response.status, 200);
      assert.equal(result.status, 'pass');
      assert.equal(result.engine, 'workerd');
      assert.equal(result.formatVersion, 1);
      assert.equal(result.samples, 4);
      assert.equal(result.individualAndBatchEqual, true);
      assert.equal(result.outputsSumToOne, true);
      assert.equal(result.isolatedFromDOM, true);
      process.stdout.write('PASS: workerd local; modelo JSON v1, inferência, lote, isolamento e validação\n');
      passed = true;
      break;
    } catch (error) {
      lastError = String(error.message || error);
      await sleep(350);
    }
  }
  if (!passed) throw new Error(`Worker não passou: ${lastError}; ${logs}`);
} finally {
  if (child.exitCode === null && child.signalCode === null) {
    const exiting = once(child, 'exit');
    child.kill('SIGTERM');
    await Promise.race([exiting, sleep(3000)]);
    if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL');
  }
}
