import test from 'node:test';
import assert from 'node:assert/strict';
import { loadModel, predict, predictBatch, serializeModel } from '../src/index.js';

function model(activations = ['tanh', 'sigmoid']) {
  return loadModel({ formatVersion: 1, layers: [
    { inputSize: 3, outputSize: 4, activation: activations[0],
      weights: [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.3, 0.7, 0.2, 0.15, -0.9, 0.55],
      biases: [0.11, -0.2, 0.03, 0.5] },
    { inputSize: 4, outputSize: 3, activation: activations[1],
      weights: [0.3, -0.2, 0.1, 0.8, -0.4, 0.7, 0.1, -0.2, 0.9, -0.4, 0.15, 0.23],
      biases: [0, 0.2, -0.1] }
  ]});
}

test('batch retorna saídas float32 independentes, idênticas a predict', () => {
  for (const activations of [['tanh', 'sigmoid'], ['relu', 'softmax'], ['linear', 'linear']]) {
    const network = model(activations);
    const snapshot = serializeModel(network);
    const vectors = [[0, 0, 0], new Float32Array([0.2, -0.3, 1.5]),
      [1e-20, 0.01, -3], [0.5, -0.6, 0.7], [0.5, -0.6, 0.7]];
    const originalVectors = vectors.map(value => [...value]);
    const results = predictBatch(network, vectors);
    assert.equal(results.length, vectors.length);
    for (let i = 0; i < results.length; i++) {
      assert.ok(results[i] instanceof Float32Array);
      assert.deepEqual(results[i], predict(network, vectors[i]));
      assert.notStrictEqual(results[i], results[(i + 1) % results.length]);
    }
    results[0][0] = 999;
    assert.deepEqual(predictBatch(network, vectors)[0], predict(network, vectors[0]));
    assert.deepEqual(vectors.map(value => [...value]), originalVectors);
    assert.deepEqual(serializeModel(network), snapshot);
  }
});

test('batch aceita vazio sem executar inferência e valida envelope', () => {
  const network = model();
  assert.deepEqual(predictBatch(network, []), []);
  for (const invalid of [null, 1, 'inputs', new Float32Array([1]), {}, [[]], [[1, 2]],
    [[1, 2, NaN]], [[1, 2, Infinity]], [[1, 2, 1e100]]]) {
    assert.throws(() => predictBatch(network, invalid));
  }
  assert.throws(() => predictBatch({}, [[1, 2, 3]]));
  assert.throws(() => predictBatch(network, new Array(4097).fill([0, 0, 0])), /limite|tamanho/i);
});

test('batch mantém semântica de erros e não retorna resultados parciais', () => {
  const network = model(['tanh', 'linear']);
  assert.throws(() => predictBatch(network, [[0, 0, 0], [0, 0, 1e100]]));
  assert.deepEqual(predictBatch(network, [[0, 0, 0]]), [predict(network, [0, 0, 0])]);
});

test('batch preserva valores nos limites e detecta overflow em camadas', () => {
  const sigmoid = loadModel({formatVersion:1,layers:[
    {inputSize:1,outputSize:1,activation:'sigmoid',weights:[1],biases:[0]}]});
  assert.deepEqual(predictBatch(sigmoid, [[-1000], [0], [1000]]).map(v => v[0]), [0, 0.5, 1]);
  const overflow = loadModel({formatVersion:1,layers:[
    {inputSize:1,outputSize:1,activation:'linear',weights:[1e30],biases:[0]}]});
  assert.throws(() => predictBatch(overflow, [[1e30]]), /numérico|finito|overflow/i);
});

test('bateladas repetidas não compartilham buffers', () => {
  const network = model();
  const a = predictBatch(network, [[1, 2, 3], [2, 3, 4]]);
  const b = predictBatch(network, [[1, 2, 3], [2, 3, 4]]);
  assert.deepEqual(a, b);
  assert.notStrictEqual(a[0], b[0]);
  a[0][0] = -12;
  assert.deepEqual(b, predictBatch(network, [[1, 2, 3], [2, 3, 4]]));
});
