import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as core from 'neural-core';
import * as direct from '../src/index.js';

const example = () => ({
  formatVersion: 1,
  layers: [{ inputSize: 2, outputSize: 1, activation: 'linear', weights: [2, 3], biases: [1] }]
});

test('o pacote preserva as três funções legadas e expõe predictBatch', () => {
  const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
  assert.equal(pkg.version, '0.1.0');
  assert.equal(pkg.type, 'module');
  assert.equal(pkg.private, true);
  assert.equal(pkg.license, 'MIT');
  assert.deepEqual(pkg.exports, { '.': './src/index.js' });
  assert.deepEqual(Object.keys(core).sort(), ['describeModel', 'getCapabilities', 'loadModel', 'predict', 'predictBatch', 'serializeModel']);
  assert.equal(core.loadModel, direct.loadModel);
  assert.equal(core.getCapabilities, direct.getCapabilities);
  assert.equal(core.describeModel, direct.describeModel);
  assert.equal(core.predict, direct.predict);
  assert.equal(core.predictBatch, direct.predictBatch);
  assert.equal(core.serializeModel, direct.serializeModel);
});

test('API v0.1.0: carregar, prever e serializar preserva a resposta', () => {
  const model = core.loadModel(example());
  assert.deepEqual(Array.from(core.predict(model, [4, 5])), [24]);
  assert.deepEqual(core.serializeModel(model), example());
  const reloaded = core.loadModel(JSON.stringify(core.serializeModel(model)));
  assert.deepEqual(Array.from(core.predict(reloaded, [4, 5])), [24]);
});

test('API pública protege pesos originais, resultado e rejeita modelos falsos', () => {
  const original = example();
  const model = core.loadModel(original);
  original.layers[0].weights[0] = 999;
  model.layers[0].weights[0] = 999;
  const result = core.predict(model, [4, 5]);
  assert.deepEqual(Array.from(result), [24]);
  result[0] = 999;
  assert.deepEqual(Array.from(core.predict(model, [4, 5])), [24]);
  assert.throws(() => core.predict({}, [4, 5]), TypeError);
  assert.throws(() => core.loadModel({ ...example(), formatVersion: 2 }), TypeError);
});
