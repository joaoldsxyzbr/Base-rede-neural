import { loadModel, predict, serializeModel } from './src/index.js';

const result = document.getElementById('result');
try {
  if (!globalThis.chrome?.runtime?.id || !location.protocol.startsWith('chrome-extension')) {
    throw new Error('Not running in a Chrome extension');
  }
  const fixtureResponse = await fetch('./fixture.json');
  if (!fixtureResponse.ok) throw new Error(`Failed to load bundled fixture (${fixtureResponse.status})`);
  const fixture = await fixtureResponse.json();
  if (fixture.fixtureVersion !== 1 || fixture.cases.length !== 4) throw new Error('Unexpected fixture');

  let checked = 0;
  for (const entry of fixture.cases) {
    const model = loadModel(entry.model);
    const restored = loadModel(serializeModel(model));
    for (const sample of entry.samples) {
      const values = predict(model, sample.input);
      const reloaded = predict(restored, sample.input);
      if (values.length !== sample.expected.length) throw new Error('Dimension mismatch');
      for (let i = 0; i < values.length; i += 1) {
        const tolerance = 1e-5 + 1e-5 * Math.abs(sample.expected[i]);
        if (!Number.isFinite(values[i]) ||
          Math.abs(values[i] - sample.expected[i]) > tolerance ||
          values[i] !== reloaded[i]) throw new Error(`Parity mismatch: ${entry.name}`);
      }
      checked++;
    }
  }
  if (checked !== 19) throw new Error(`Expected 19 predictions, got ${checked}`);

  const worker = await Promise.race([
    chrome.runtime.sendMessage({ type: 'neural-core-p5' }),
    new Promise((_, reject) => setTimeout(() => reject(new Error('Service worker did not respond')), 15000))
  ]);
  if (!worker?.ok || worker.output !== 24 || !worker.isWorker) {
    throw new Error('Background service worker failed neural inference');
  }
  result.textContent = `PASS: MV3 popup (${checked} cases) and service worker`;
  result.dataset.status = 'pass';
} catch (error) {
  result.textContent = `FAIL: ${error?.message || error}`;
  result.dataset.status = 'fail';
}
