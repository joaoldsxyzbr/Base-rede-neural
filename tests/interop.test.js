import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { loadModel, predict, serializeModel, getCapabilities, describeModel } from '../src/index.js';

const source = () => ({formatVersion: 1, layers: [
  {inputSize: 2, outputSize: 3, activation: 'relu',
    weights: [1,2,-1,0,0.5,0.75], biases:[0,0.1,0]},
  {inputSize: 3, outputSize: 2, activation:'softmax',
    weights:[1,0,-1,-1,0,1], biases:[0.2,-0.2]}
]});

test('P14: capacidades descrevem somente operadores/formatos realmente suportados', () => {
  const capabilities = getCapabilities();
  assert.deepEqual(capabilities, {
    formatVersions: [1], operators: ['dense'],
    activations: ['linear', 'relu', 'sigmoid', 'softmax', 'tanh'],
    limits: {maxLayers: 16, maxWidth: 4096, maxParameters: 250000}
  });
  assert.deepEqual(getCapabilities(), capabilities);
  assert.ok(Object.isFrozen(capabilities));
  assert.ok(Object.isFrozen(capabilities.activations));
  assert.ok(Object.isFrozen(capabilities.limits));
  assert.throws(() => {capabilities.activations[0]='arbitrary';}, TypeError);
  assert.throws(() => {capabilities.limits.maxLayers=99;}, TypeError);
  assert.ok(!capabilities.activations.includes('eval'));
  assert.ok(!capabilities.formatVersions.includes(2));
});

test('P14: descrição não expõe pesos e explica arquitetura/contagem', () => {
  const network = loadModel(source());
  const info = describeModel(network);
  assert.deepEqual(info, {
    formatVersion: 1, inputSize: 2, outputSize: 2,
    layerCount: 2, parameterCount: 17,
    layers: [
      {inputSize:2,outputSize:3,activation:'relu',parameterCount:9},
      {inputSize:3,outputSize:2,activation:'softmax',parameterCount:8}
    ]
  });
  assert.deepEqual(JSON.parse(JSON.stringify(info)),info);
  assert.ok(Object.isFrozen(info) && Object.isFrozen(info.layers) && Object.isFrozen(info.layers[0]));
  assert.throws(()=>{info.layers[0].activation='eval';},TypeError);
  assert.throws(()=>{info.layers.push({});},TypeError);
  assert.equal(Object.hasOwn(info,'weights'),false);
  assert.equal(JSON.stringify(info).includes('weights'),false);
  assert.deepEqual(serializeModel(loadModel(serializeModel(network))),serializeModel(network));
  assert.deepEqual(Array.from(predict(network,[0.5,-1])),Array.from(predict(loadModel(source()),[0.5,-1])));
});

test('P14: apenas modelos carregados podem ser descritos; limite 250k válido', () => {
  assert.throws(()=>describeModel({formatVersion:1,layers:[]}), TypeError);
  assert.throws(()=>describeModel(null), TypeError);
  const one=loadModel({formatVersion:1,layers:[{
    inputSize: 1,outputSize:1,activation:'linear',weights:[3],biases:[2]
  }]});
  assert.deepEqual(describeModel(one),{
    formatVersion:1,inputSize:1,outputSize:1,layerCount:1,parameterCount:2,
    layers:[{inputSize:1,outputSize:1,activation:'linear',parameterCount:2}]
  });
  assert.throws(()=>loadModel({...source(),formatVersion:2}),/versão|inválido/i);
  assert.throws(()=>loadModel({...source(),layers:[{...source().layers[0],activation:'unsupported'}]}),/ativação|inválida/i);
});

test('P14: Python e JavaScript negociam contratos e descrições iguais', {
  skip: !process.env.NEURAL_CORE_INTEROP_FIXTURE
}, () => {
  const fixture = JSON.parse(readFileSync(process.env.NEURAL_CORE_INTEROP_FIXTURE, 'utf8'));
  assert.equal(fixture.conformanceVersion,1);
  assert.deepEqual(getCapabilities(),fixture.capabilities);
  assert.ok(fixture.models.length >= 3);
  for(const {name, model, description} of fixture.models){
    const loaded = loadModel(model);
    assert.deepEqual(describeModel(loaded),description, name);
    assert.deepEqual(serializeModel(loaded),model,name);
    assert.deepEqual(describeModel(loadModel(JSON.stringify(model))),description,name);
  }
});
