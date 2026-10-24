/** P12: comparação informativa dentro do mesmo processo, sem limite de velocidade. */
import { performance } from 'node:perf_hooks';
import assert from 'node:assert/strict';
import { loadModel, predict, predictBatch } from '../src/index.js';

function build(widths, lastActivation = 'sigmoid') {
  const layers = widths.slice(1).map((outputSize, index) => {
    const inputSize = widths[index];
    return {
      inputSize, outputSize,
      activation: index === widths.length - 2 ? lastActivation : 'tanh',
      weights: Array.from({length: inputSize * outputSize}, (_, i) => ((i % 17) - 8) * 0.004),
      biases: Array(outputSize).fill(0.01)
    };
  });
  return loadModel({formatVersion: 1, layers});
}

function median(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

const configs = [
  {name:'small', widths:[8,16,4], count:64, repetitions:100},
  {name:'medium', widths:[32,64,16], count:64, repetitions:50},
  {name:'large', widths:[128,128,32], count:32, repetitions:25},
  {name:'deep', widths:[16,48,32,24,8], count:64, repetitions:40},
  {name:'softmax', widths:[16,32,4], count:64, repetitions:50, activation:'softmax'}
];

function execute(config) {
  const model = build(config.widths, config.activation);
  const inputs = Array.from({length: config.count}, (_, i) =>
    Float32Array.from({length: config.widths[0]}, (_, j) => (i + j % 7) * 0.003));
  const loop = () => inputs.map(x => predict(model, x));
  const batch = () => predictBatch(model, inputs);

  const expected = loop(), actual = batch();
  assert.deepEqual(actual, expected, `${config.name}: lote divergiu do caminho individual`);
  const measure = fn => {
    const start = performance.now();
    let checksum = 0;
    for (let i = 0; i < config.repetitions; i++) {
      const outputs = fn();
      checksum += outputs[i % outputs.length][0];
    }
    if (!Number.isFinite(checksum)) throw new Error('checksum inválido');
    return (performance.now() - start) * 1000 / (config.count * config.repetitions);
  };
  for (let warm = 0; warm < 5; warm++) { loop(); batch(); }
  const single = [], batched = [];
  for (let i = 0; i < 5; i++) {
    if (i % 2) { batched.push(measure(batch)); single.push(measure(loop)); }
    else { single.push(measure(loop)); batched.push(measure(batch)); }
  }
  const a = median(single), b = median(batched);
  return {
    name: config.name, topology: config.widths,
    inputsPerBatch: config.count, repetitionsPerMeasurement: config.repetitions,
    singleMicrosecondsPerInput: Number(a.toFixed(3)),
    batchedMicrosecondsPerInput: Number(b.toFixed(3)),
    speedupObserved: Number((a / b).toFixed(3)),
    samples: {
      single: single.map(x => Number(x.toFixed(3))),
      batched: batched.map(x => Number(x.toFixed(3)))
    }, exactParity: true
  };
}

process.stdout.write(JSON.stringify({
  benchmarkVersion: 1,
  runtime: {node: process.version, platform: process.platform, arch: process.arch},
  methodology: 'Same process; warmup + five alternating measurements; median; no hard gate; volatile CI timings',
  cases: configs.map(execute)
}, null, 2) + '\n');
