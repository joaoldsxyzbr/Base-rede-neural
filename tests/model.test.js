import test from 'node:test';
import assert from 'node:assert/strict';
import { loadModel, serializeModel } from '../src/index.js';

const valid = () => ({
  formatVersion: 1,
  layers: [{ inputSize: 2, outputSize: 1, activation: 'linear', weights: [2, -3], biases: [0.5] }]
});

test('carrega modelo independente e valida estrutura', () => {
  const raw = valid();
  const model = loadModel(raw);
  raw.layers[0].weights[0] = 99;
  assert.equal(model.layers[0].weights[0], 2);
  assert.ok(model.layers[0].weights instanceof Float32Array);
  assert.equal(model.inputSize, 2);
  assert.equal(model.outputSize, 1);
});

test('serialização preserva pesos e formato universal', () => {
  const model = loadModel(valid());
  const serialized = serializeModel(model);
  assert.deepEqual(serialized, valid());
  assert.deepEqual(serializeModel(loadModel(JSON.stringify(serialized))), valid());
});

test('rejeita formato desconhecido, vazio, desconexo ou camadas ausentes', () => {
  const cases = [
    { ...valid(), formatVersion: 2 },
    { ...valid(), layers: [] },
    { ...valid(), layers: [valid().layers[0], { inputSize: 3, outputSize: 1, activation: 'linear', weights: [1, 1, 1], biases: [0] }] },
    { formatVersion: 1 }
  ];
  for (const candidate of cases) assert.throws(() => loadModel(candidate));
});

test('rejeita tamanho incorreto, parâmetros não finitos e ativação inválida', () => {
  for (const patch of [
    { weights: [1] }, { biases: [] }, { weights: [1, Infinity] },
    { weights: [1, NaN] }, { weights: [1, 1e50] }, { activation: 'eval' },
    { outputSize: -1 }, { inputSize: 0 }, { inputSize: 1.5 }
  ]) {
    const model = valid();
    Object.assign(model.layers[0], patch);
    assert.throws(() => loadModel(model));
  }
});

test('rejeita modelos excessivamente grandes antes de alocar matrizes', () => {
  const model = valid();
  model.layers[0].inputSize = 5000;
  model.layers[0].outputSize = 5000;
  assert.throws(() => loadModel(model), /limite|máximo|inválido|tamanho/i);
  assert.throws(() => loadModel('x'.repeat(25_000_001)), /limite|tamanho/i);
});

test('expor camadas não permite alterar pesos ou vieses usados na inferência', async () => {
  const { predict } = await import('../src/index.js');
  const model = loadModel(valid());
  const original = Array.from(predict(model, [3, 4]));
  const exported = serializeModel(model);

  const snapshot = model.layers;
  snapshot[0].weights[0] = 100;
  snapshot[0].biases[0] = 200;
  new DataView(snapshot[0].weights.buffer).setFloat32(4, 500, true);
  assert.throws(() => { snapshot.push(snapshot[0]); }, TypeError);
  assert.throws(() => { model.inputSize = 999; }, TypeError);

  assert.deepEqual(Array.from(predict(model, [3, 4])), original);
  assert.deepEqual(serializeModel(model), exported);
  assert.equal(model.layers[0].weights[0], 2);
  assert.equal(model.layers[0].biases[0], 0.5);
});

test('modelos distintos não compartilham pesos mesmo quando a origem é a mesma', async () => {
  const { predict } = await import('../src/index.js');
  const source = valid();
  const first = loadModel(source);
  const second = loadModel(source);
  first.layers[0].weights[0] = 99;
  assert.deepEqual(Array.from(predict(first, [1, 1])), Array.from(predict(second, [1, 1])));
  assert.equal(serializeModel(second).layers[0].weights[0], 2);
});

test('rejeita modelos malformados, grandes e tipos incompatíveis', () => {
  assert.throws(() => loadModel('{invalid'));
  assert.throws(() => loadModel(null));
  assert.throws(() => loadModel([]));
  assert.throws(() => loadModel({formatVersion: 1, layers: [null]}));
  assert.throws(() => loadModel({formatVersion: 1, layers: new Array(17).fill(valid().layers[0])}));
  assert.throws(() => loadModel({...valid(), layers: [{...valid().layers[0], biases: [NaN]}]}));
  assert.throws(() => loadModel({...valid(), layers: [{...valid().layers[0], weights: new Float32Array([2, -3])}]}));
  assert.throws(() => loadModel({...valid(), layers: [{...valid().layers[0], biases: [1e100]}]}));
});
