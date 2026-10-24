/** Teste isolado do runtime real workerd; não integra o motor Neural Core. */
import { loadModel, predict, predictBatch, serializeModel, getCapabilities, describeModel } from '../src/index.js';

export default {
  async fetch() {
    const model = loadModel({ formatVersion: 1, layers: [
      { inputSize: 2, outputSize: 3, activation: 'relu',
        weights: [1, -1, 2, 0, 0, 2], biases: [0, 0, 0] },
      { inputSize: 3, outputSize: 2, activation: 'softmax',
        weights: [1, 0, -1, -1, 0, 1], biases: [0, 0] }
    ] });
    const info = describeModel(model);
    const support = getCapabilities();
    if (!support.formatVersions.includes(1) || !support.activations.includes('softmax') ||
        info.layerCount !== 2 || info.parameterCount !== 17) {
      return Response.json({ status: 'fail', reason: 'capability-contract' }, { status: 500 });
    }
    const inputs = [[1, 2], [2, 1], [0, 0], [-1, 1]];
    const single = inputs.map(input => Array.from(predict(model, input)));
    const batch = predictBatch(model, inputs).map(value => Array.from(value));
    if (JSON.stringify(single) !== JSON.stringify(batch)) {
      return Response.json({ status: 'fail', reason: 'batch-parity' }, { status: 500 });
    }
    const serialized = serializeModel(model);
    const restored = loadModel(serialized);
    if (JSON.stringify(Array.from(predict(restored, inputs[0]))) !== JSON.stringify(single[0])) {
      return Response.json({ status: 'fail', reason: 'serialization' }, { status: 500 });
    }
    let rejected = false;
    try { loadModel({ ...serialized, formatVersion: 2 }); } catch (error) {
      rejected = error instanceof TypeError;
    }
    if (!rejected) {
      return Response.json({ status: 'fail', reason: 'invalid-model' }, { status: 500 });
    }
    const independent = predictBatch(model, inputs);
    independent[0][0] = 99;
    if (predict(model, inputs[0])[0] === 99 || independent[1][0] === 99) {
      return Response.json({ status: 'fail', reason: 'aliasing' }, { status: 500 });
    }
    return Response.json({
      status: 'pass',
      engine: 'workerd',
      formatVersion: serialized.formatVersion,
      samples: inputs.length,
      individualAndBatchEqual: true,
      outputsSumToOne: single.every(row => Math.abs(row.reduce((a,b) => a+b, 0) - 1) < 1e-6),
      isolatedFromDOM: typeof document === 'undefined'
    });
  }
};
