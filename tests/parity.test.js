import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { loadModel, predict, serializeModel } from '../src/index.js';

const fixturePath = process.env.NEURAL_CORE_PARITY_FIXTURE
  || fileURLToPath(new URL('./fixtures/parity.json', import.meta.url));
const fixture = JSON.parse(readFileSync(fixturePath, 'utf8'));
const ABSOLUTE_TOLERANCE = 1e-5;
const RELATIVE_TOLERANCE = 1e-5;

function equalWithinTolerance(actual, expected, description) {
  assert.ok(Number.isFinite(actual) && Number.isFinite(expected), `${description}: valor não finito`);
  const difference = Math.abs(actual - expected);
  const tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * Math.abs(expected);
  assert.ok(difference <= tolerance,
    `${description}: Python=${expected}, JavaScript=${actual}, diferença=${difference}, tolerância=${tolerance}`);
}

test('fixture de paridade tem estrutura e variedade suficientes', () => {
  assert.equal(fixture.fixtureVersion, 1);
  assert.ok(Array.isArray(fixture.cases) && fixture.cases.length >= 4);
  const activations = new Set();
  for (const candidate of fixture.cases) {
    assert.ok(candidate.name && Array.isArray(candidate.samples) && candidate.samples.length > 0);
    const model = loadModel(candidate.model);
    for (const layer of candidate.model.layers) activations.add(layer.activation);
    for (const sample of candidate.samples) {
      assert.equal(sample.input.length, model.inputSize);
      assert.equal(sample.expected.length, model.outputSize);
    }
  }
  assert.deepEqual([...activations].sort(), ['linear', 'sigmoid', 'tanh']);
});

for (const candidate of fixture.cases) {
  test(`paridade Python × JS: ${candidate.name}`, () => {
    const model = loadModel(candidate.model);
    const restored = loadModel(JSON.stringify(serializeModel(model)));
    for (const [index, sample] of candidate.samples.entries()) {
      const jsOutput = predict(model, sample.input);
      const reloadedOutput = predict(restored, sample.input);
      assert.equal(jsOutput.length, sample.expected.length);
      for (let i = 0; i < jsOutput.length; i += 1) {
        const label = `${candidate.name}, amostra ${index}, saída ${i}`;
        equalWithinTolerance(jsOutput[i], sample.expected[i], label);
        assert.equal(reloadedOutput[i], jsOutput[i], `${label}: serialização alterou previsão`);
      }
    }
  });
}

test('teste de paridade detecta divergência acima da tolerância', () => {
  assert.throws(() => equalWithinTolerance(0.1, 0.2, 'divergência proposital'), /diferença=/);
});
