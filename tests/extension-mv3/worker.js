import { loadModel, predict } from './src/index.js';

// Listener is registered synchronously, as required by MV3 service workers.
chrome.runtime.onMessage.addListener((message, _sender, respond) => {
  if (message?.type !== 'neural-core-p5') return false;
  try {
    const model = loadModel({
      formatVersion: 1,
      layers: [{ inputSize: 2, outputSize: 1, activation: 'linear', weights: [2, 3], biases: [1] }]
    });
    const [output] = predict(model, [4, 5]);
    respond({ ok: true, output, isWorker: typeof document === 'undefined' });
  } catch (error) {
    respond({ ok: false, error: String(error?.message || error) });
  }
  return false;
});
