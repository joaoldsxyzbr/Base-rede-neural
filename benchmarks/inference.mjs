/** Benchmark informativo: não introduzir limites rígidos em runners compartilhados. */
import { performance } from 'node:perf_hooks';
import { loadModel, predict } from '../src/index.js';

const scenarios = [
  { name: 'small', widths: [8, 16, 4], iterations: 6000 },
  { name: 'medium', widths: [32, 64, 16], iterations: 1600 },
  { name: 'large', widths: [128, 128, 32], iterations: 350 }
];

function rawModel(widths) {
  return {
    formatVersion: 1,
    layers: widths.slice(1).map((outputSize, index) => {
      const inputSize = widths[index];
      return {
        inputSize, outputSize,
        activation: index === widths.length - 2 ? 'sigmoid' : 'tanh',
        weights: Array.from({ length: inputSize * outputSize }, (_, i) => ((i % 17) - 8) * 0.004),
        biases: Array.from({ length: outputSize }, () => 0.01)
      };
    })
  };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function run(scenario) {
  const raw = rawModel(scenario.widths);
  const encoded = JSON.stringify(raw);
  const before = process.memoryUsage();
  const loadStart = performance.now();
  const model = loadModel(encoded);
  const loadMs = performance.now() - loadStart;
  const loaded = process.memoryUsage();
  const input = new Float32Array(scenario.widths[0]).fill(0.2);
  let checksum = 0;
  for (let i = 0; i < 200; i += 1) checksum += predict(model, input)[0];

  const samples = [];
  for (let repetition = 0; repetition < 5; repetition += 1) {
    const start = performance.now();
    for (let i = 0; i < scenario.iterations; i += 1) {
      checksum += predict(model, input)[0];
    }
    samples.push((performance.now() - start) * 1000 / scenario.iterations);
  }
  if (!Number.isFinite(checksum)) throw new Error('Benchmark gerou resultado não finito');
  const parameters = raw.layers.reduce((sum, layer) => sum + layer.weights.length + layer.biases.length, 0);
  return {
    name: scenario.name,
    layers: scenario.widths,
    parameters,
    weightBytesFloat32: parameters * 4,
    jsonBytes: Buffer.byteLength(encoded),
    loadMs: Number(loadMs.toFixed(3)),
    medianInferenceMicroseconds: Number(median(samples).toFixed(3)),
    samplesMicroseconds: samples.map(value => Number(value.toFixed(3))),
    observedHeapDeltaBytes: loaded.heapUsed - before.heapUsed,
    observedRssDeltaBytes: loaded.rss - before.rss
  };
}

const report = {
  benchmarkVersion: 1,
  runtime: { node: process.version, platform: process.platform, arch: process.arch },
  methodology: '200 warmups; 5 repeated batches per case, median microseconds/prediction; no hard performance gate',
  cases: scenarios.map(run)
};
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
