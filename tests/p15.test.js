import test from 'node:test';
import assert from 'node:assert/strict';
import {loadModel,predict,predictBatch,serializeModel,describeModel,getCapabilities} from '../src/index.js';

const raw = () => ({ formatVersion:1, layers:[{
  inputSize:2,outputSize:2,activation:'softmax',
  weights:[1,-1,-1,1],biases:[0.125,-0.125]
}]});

function nextRandom(seed) {
  let state=seed>>>0;
  return () => {state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296;};
}

test('P15: operador v1 explícito dense é aceito, e operador desconhecido é rejeitado', () => {
  const v1=raw();
  const annotated={...v1,layers:[{...v1.layers[0],operator:'dense'}]};
  const loaded=loadModel(annotated);
  assert.deepEqual(serializeModel(loaded),v1);
  assert.deepEqual(predict(loaded,[0.3,0.1]), predict(loadModel(v1),[0.3,0.1]));
  for(const operator of ['conv','attention','eval', '', null, 1, ['dense']]) {
    const invalid={...v1,layers:[{...v1.layers[0],operator}]};
    assert.throws(()=>loadModel(invalid), /operador|operator/i);
  }
  const decorated={...v1,description:'metadata',layers:[{...v1.layers[0],description:'extra'}]};
  assert.deepEqual(serializeModel(loadModel(decorated)),v1);
  assert.deepEqual(getCapabilities().operators,['dense']);
});

test('P15: teste gerativo determinístico verifica serialização e lote em topologias variadas', () => {
  const rand=nextRandom(0x5a17);
  const activations=['linear','tanh','sigmoid','relu','softmax'];
  for(let iter=0;iter<220;iter++){
    const depth=1+Math.floor(rand()*4);
    let size=1+Math.floor(rand()*8);
    const layers=[];
    for(let k=0;k<depth;k++){
      const width=2+Math.floor(rand()*9);
      const activation=activations[Math.floor(rand()*(k===depth-1?5:4))];
      const weights=Array.from({length:size*width},()=>Math.fround((rand()-0.5)*2.5));
      const biases=Array.from({length:width},()=>Math.fround((rand()-0.5)*0.2));
      layers.push({inputSize:size,outputSize:width,activation,weights,biases});
      size=width;
    }
    const source={formatVersion:1,layers};
    const model=loadModel(source);
    const roundtrip=loadModel(JSON.stringify(serializeModel(model)));
    const inputs=Array.from({length:6},()=>Array.from({length:layers[0].inputSize},
      ()=>Math.fround((rand()-0.5)*5)));
    const batch=predictBatch(model,inputs);
    assert.equal(batch.length,inputs.length);
    for(let j=0;j<inputs.length;j++){
      assert.deepEqual(batch[j],predict(model,inputs[j]),'lote '+iter+'/'+j);
      assert.deepEqual(batch[j],predict(roundtrip,inputs[j]),'serialização '+iter+'/'+j);
      assert.ok(batch[j].every(Number.isFinite));
    }
    const description=describeModel(model);
    assert.equal(description.parameterCount,layers.reduce((n,l)=>n+l.weights.length+l.biases.length,0));
  }
});

test('P15: mutações inválidas repetidas nunca contaminam modelos carregados',()=>{
  const model=loadModel(raw()), baseline=serializeModel(model);
  const invalids=[
    {...raw(),layers:[{...raw().layers[0],weights:[1,2,NaN,3]}]},
    {...raw(),layers:[{...raw().layers[0],weights:[1,2,Infinity,3]}]},
    {...raw(),layers:[{...raw().layers[0],weights:[1,2,1e50,3]}]},
    {...raw(),layers:[{...raw().layers[0],outputSize:4097}]},
    {...raw(),layers:[{...raw().layers[0],activation:'unknown'}]},
    {...raw(),layers:[{...raw().layers[0],operator:'conv'}]},
    {...raw(),layers:[{...raw().layers[0],biases:[0,0,0]}]},
    {...raw(),formatVersion:2}
  ];
  for(let repeat=0;repeat<30;repeat++){
    for(const input of invalids) assert.throws(()=>loadModel(input));
    assert.deepEqual(serializeModel(model),baseline);
    assert.deepEqual(predictBatch(model,[[1,0],[0,1]]),[[1,0],[0,1]].map(x=>predict(model,x)));
  }
});
