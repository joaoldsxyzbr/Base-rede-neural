import { loadModel, predict, serializeModel } from '../src/index.js';

// Exemplo genérico. Os parâmetros abaixo não foram treinados.
const model = loadModel({
  formatVersion: 1,
  layers: [{ inputSize: 2, outputSize: 1, activation: 'linear', weights: [2, 3], biases: [1] }]
});
console.log('Previsão:', Array.from(predict(model, [4, 5])));
console.log('Modelo exportado:', JSON.stringify(serializeModel(model)));
