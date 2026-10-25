/** P15: stress do lote e memória observada; não impõe limiares de heap variáveis. */
import assert from 'node:assert/strict';
import {loadModel,predict,predictBatch,serializeModel} from '../src/index.js';

const inputSize=16, hidden=24, outputSize=8;
const model=loadModel({formatVersion:1,layers:[
  {inputSize,outputSize:hidden,activation:'relu',
   weights:Array.from({length:inputSize*hidden},(_,i)=>((i%13)-6)/37),
   biases:Array(hidden).fill(0)},
  {inputSize:hidden,outputSize,activation:'softmax',
   weights:Array.from({length:hidden*outputSize},(_,i)=>((i%17)-8)/43),
   biases:Array(outputSize).fill(0)}
]});
const samples=Array.from({length:4096},(_,i)=>
  Float32Array.from({length:inputSize},(_,j)=>Math.sin((i+1)*(j+1)*0.02)));
const before=JSON.stringify(serializeModel(model));
const memory=()=>({rss:process.memoryUsage().rss,heapUsed:process.memoryUsage().heapUsed});
global.gc?.();
const baseline=memory();
let checksum=0;
const t0=performance.now();
for(let round=0;round<12;round++){
  const outputs=predictBatch(model,samples);
  assert.equal(outputs.length,4096);
  for(const index of [0,1,500,2048,4095]) {
    const expected=predict(model,samples[index]);
    assert.deepEqual(outputs[index],expected);
    checksum+=outputs[index][0];
  }
  assert.notStrictEqual(outputs[0],outputs[1]);
  outputs[0][0]=-1;
  assert.deepEqual(predict(model,samples[0]),predictBatch(model,[samples[0]])[0]);
}
const elapsed=performance.now()-t0;
global.gc?.();
const ending=memory();
assert.equal(JSON.stringify(serializeModel(model)),before);
assert.ok(Number.isFinite(checksum));
process.stdout.write(JSON.stringify({
  benchmarkVersion:1, type:'diagnostic-memory-stress',
  topology:[inputSize,hidden,outputSize],
  batchSize:4096, batches:12, samplesProcessed:4096*12,
  durationMs:Math.round(elapsed),
  memoryBytesBefore:baseline,memoryBytesAfter:ending,
  note:'RSS/heap delta é observacional; GC/JIT/runner influenciam e não é limite de aprovação'
},null,2)+'\n');
