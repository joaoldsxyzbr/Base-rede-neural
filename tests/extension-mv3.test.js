import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const base = new URL('./extension-mv3/', import.meta.url);
const read = name => readFileSync(new URL(name, base), 'utf8');

test('Manifest V3 has minimal scope, module service worker and no site access', () => {
  const manifest = JSON.parse(read('manifest.json'));
  assert.equal(manifest.manifest_version, 3);
  assert.equal(manifest.background.type, 'module');
  assert.equal(manifest.background.service_worker, 'worker.js');
  assert.equal(manifest.action.default_popup, 'popup.html');
  assert.ok(manifest.key.length > 100);
  for (const permission of ['permissions', 'host_permissions', 'content_scripts', 'externally_connectable', 'web_accessible_resources']) {
    assert.equal(Object.hasOwn(manifest, permission), false, `${permission} should not be granted`);
  }
});

test('Both contexts import the original Core and do not load remote scripts', () => {
  const html = read('popup.html');
  const popup = read('popup.js');
  const worker = read('worker.js');
  assert.match(html, /type="module" src="popup.js"/);
  assert.match(popup, /from '\.\/src\/index\.js'/);
  assert.match(worker, /from '\.\/src\/index\.js'/);
  assert.match(worker, /chrome\.runtime\.onMessage\.addListener/);
  for (const source of [html, popup, worker]) {
    assert.doesNotMatch(source, /https?:\/\//);
  }
});
