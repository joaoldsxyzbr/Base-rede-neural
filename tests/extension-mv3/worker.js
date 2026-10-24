import { loadModel, predict, predictBatch, getCapabilities, describeModel } from './src/index.js';

// Listener is registered synchronously, as required by MV3 service workers.
chrome.runtime.onMessage.addListener((message, _sender, respond) => {
  if (message?.type !== 'neural-core-p5') return false;
  try {
    const model = loadModel({
      formatVersion: 1,
      layers: [{ inputSize: 2, outputSize: 1, activation: 'linear', weights: [2, 3], biases: [1] }]
    });
    if (!getCapabilities().operators.includes('dense') || describeModel(model).parameterCount !== 3)
      throw new Error('Contract mismatch in MV3 service worker');
    const [output] = predict(model, [4, 5]);
    const batched = predictBatch(model, [[4, 5], [1, 2]]);
    if (batched[0][0] !== output || batched[1][0] !== 9) throw new Error('Batch diverged in worker');
    respond({ ok: true, output, isWorker: typeof document === 'undefined' });
  } catch (error) {
    respond({ ok: false, error: String(error?.message || error) });
  }
  return false;
});
