import test from 'node:test';
import assert from 'node:assert/strict';
import { loadModel, predict, serializeModel } from '../src/index.js';

test('relu não altera o formato v1 e não cria valores negativos', () => {
  const model = loadModel({formatVersion:1,layers:[{
    inputSize:2,outputSize:2,activation:'relu',
    weights:[1,-2,-1,-1],biases:[0.5,0]
  }]});
  assert.deepEqual(Array.from(predict(model,[3,1])),[1.5,0]);
  assert.equal(serializeModel(model).layers[0].activation,'relu');
});

test('softmax é estável e produz probabilidades normalizadas', () => {
  const source = {formatVersion:1,layers:[{
    inputSize:1,outputSize:3,activation:'softmax',
    weights:[1000,-1000,999],biases:[0,0,0]
  }]};
  const model = loadModel(source);
  const output = predict(model,[1]);
  assert.ok(output[0] > 0.73 && output[0] < 0.74);
  assert.equal(output[1],0);
  assert.ok(Math.abs(Array.from(output).reduce((a,b)=>a+b,0)-1)<1e-6);
  assert.deepEqual(Array.from(predict(loadModel(serializeModel(model)),[1])),Array.from(output));
});

test('softmax rejeitado em camadas intermediárias e saída unitária', () => {
  const hidden={inputSize:2,outputSize:2,activation:'softmax',weights:[1,0,0,1],biases:[0,0]};
  const last={inputSize:2,outputSize:2,activation:'linear',weights:[1,0,0,1],biases:[0,0]};
  assert.throws(()=>loadModel({formatVersion:1,layers:[hidden,last]}));
  assert.throws(()=>loadModel({formatVersion:1,layers:[{...hidden,outputSize:1,weights:[1,0],biases:[0]}]}));
});
