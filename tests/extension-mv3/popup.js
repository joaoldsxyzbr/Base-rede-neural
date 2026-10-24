import { loadModel, predict, predictBatch, serializeModel, getCapabilities, describeModel } from './src/index.js';

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
    const capabilities = getCapabilities();
    const description = describeModel(model);
    if (!capabilities.formatVersions.includes(entry.model.formatVersion) ||
        description.parameterCount !== entry.model.layers.reduce((sum, layer) => sum + layer.weights.length + layer.biases.length, 0))
      throw new Error('Unsupported model contract in MV3 popup');
    const restored = loadModel(serializeModel(model));
    const batched = predictBatch(model, entry.samples.map(sample => sample.input));
    for (const [index, sample] of entry.samples.entries()) {
      const values = predict(model, sample.input);
      const reloaded = predict(restored, sample.input);
      if (values.length !== sample.expected.length) throw new Error('Dimension mismatch');
      for (let i = 0; i < values.length; i += 1) {
        const tolerance = 1e-5 + 1e-5 * Math.abs(sample.expected[i]);
        if (!Number.isFinite(values[i]) ||
          Math.abs(values[i] - sample.expected[i]) > tolerance ||
          values[i] !== reloaded[i] || values[i] !== batched[index][i]) throw new Error(`Parity mismatch: ${entry.name}`);
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
