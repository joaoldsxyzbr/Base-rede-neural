import test from 'node:test';
import assert from 'node:assert/strict';
import { loadModel, predict } from '../src/index.js';

function close(actual, expected, tolerance = 1e-6) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} deveria ser ${expected}`);
}

test('rede linear usa matriz orientada por neurônio de saída', () => {
  const model = loadModel({ formatVersion: 1, layers: [{
    inputSize: 2, outputSize: 2, activation: 'linear',
    weights: [2, 3, -1, 4], biases: [0.5, 1]
  }] });
  const output = predict(model, [1, 2]);
  assert.ok(output instanceof Float32Array);
  close(output[0], 8.5);
  close(output[1], 8);
});

test('sigmoid e tanh passam pelas duas camadas corretamente', () => {
  const model = loadModel({formatVersion: 1, layers: [
    {inputSize: 2, outputSize: 2, activation: 'tanh', weights: [1, 0, 0, 1], biases: [0, 0]},
    {inputSize: 2, outputSize: 1, activation: 'sigmoid', weights: [1, -1], biases: [0.25]}
  ]});
  const result = predict(model, [0.75, -0.25]);
  const expected = 1 / (1 + Math.exp(-(Math.tanh(0.75) - Math.tanh(-0.25) + 0.25)));
  close(result[0], expected);
  assert.deepEqual(Array.from(predict(model, [0.75, -0.25])), Array.from(result));
});

test('rejeita entradas incompletas, excedentes ou não finitas', () => {
  const model = loadModel({formatVersion: 1, layers: [{inputSize:2, outputSize:1, activation:'linear', weights:[1,2], biases:[0]}]});
  for (const x of [[1], [1,2,3], [NaN,0], [Infinity,1], [1e50,1], '1,2']) {
    assert.throws(() => predict(model, x));
  }
  assert.deepEqual(Array.from(predict(model, new Float32Array([1, 2]))), [5]);
});

test('inferência não muda pesos e detecta overflow numérico', () => {
  const model = loadModel({formatVersion: 1,layers:[{inputSize:1,outputSize:1,activation:'linear',weights:[1e30],biases:[0]}]});
  assert.throws(() => predict(model, [1e30]), /finito|numérico|overflow/i);
  assert.equal(model.layers[0].weights[0], Math.fround(1e30));
});
