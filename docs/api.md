# Contrato de API — Neural Core 0.1.0

## JavaScript público

Importar `loadModel`, `predict` e `serializeModel` de `src/index.js` (ES Modules). Dentro do repositório Node.js, a importação `neural-core` resolve para o mesmo módulo; não há publicação npm.

- **`loadModel(source)`** recebe string JSON ou objeto de dados v1; valida formato, dimensões, ativações e parâmetros e carrega os pesos como `Float32Array`.
- **`predict(model, input)`** recebe um modelo já carregado e um `Array<number>` ou `Float32Array` da dimensão correta, devolvendo novo `Float32Array`; não modifica pesos.
- **`serializeModel(model)`** devolve **um novo objeto JSON-compatível**. Use `JSON.stringify` para obter uma string.

```javascript
import { loadModel, predict, serializeModel } from './src/index.js';
const model = loadModel({ formatVersion: 1, layers: [
  { inputSize: 2, outputSize: 1, activation: 'linear', weights: [2, 3], biases: [1] }
] });
console.log(predict(model, [4, 5])); // Float32Array([24])
const saved = JSON.stringify(serializeModel(model));
```

## Modelo portável JSON v1

```json
{ "formatVersion": 1, "layers": [
  { "inputSize": 2, "outputSize": 1, "activation": "sigmoid",
    "weights": [0.25, -0.5], "biases": [0.1] }
] }
```

Camadas densas em ordem, pesos achatados por neurônio de saída `weights[output][input]` e vieses por saída. Ativações permitidas: `linear`, `sigmoid`, `tanh`, `relu` e `softmax` (somente na última camada, com duas ou mais saídas). Camadas conectadas têm dimensões compatíveis. Limites: **16 camadas**, **4.096 unidades por dimensão**, **250.000 parâmetros totais** (pesos + vieses), **25 milhões de caracteres para strings JSON**. O limite de caracteres não se aplica a objetos já materializados. O consumo total de memória depende do ambiente.

Dados não numéricos, `NaN`, infinito, valores que excedem float32, dimensões e versões inválidas são rejeitados; valores muito pequenos podem arredondar para zero. `formatVersion` precisa ser exatamente `1`. Campos desconhecidos não pertencem ao contrato e podem ser ignorados. O serializador em JS elimina campos desconhecidos; modelos v2 não são interpretados como v1. Veja [política de compatibilidade](compatibility.md).

`model.inputSize` e `model.outputSize` são estáveis. `model.layers` retorna **cópias** independentes dos pesos/vieses: lê-las pode alocar memória. Modificar essas cópias ou resultados de `predict` não altera o modelo. Erros podem ser `TypeError`, `RangeError` e `SyntaxError`; não dependa da redação das mensagens.

Modelos de fontes não confiáveis devem ser obtidos/preferencialmente analisados como JSON, **não** como objetos JS com getters ou Proxy. Esta biblioteca valida dados, mas não isola execução de objetos maliciosos já criados pelo chamador.

## Python offline

- `training.network.create_network(sizes, activations, seed=0)`: inicializa camadas densas (He uniforme para ReLU, Xavier uniforme para as demais).
- `training.network.predict(network, inputs)`: inferência Python de referência.
- `training.train.train(network, dataset, epochs=1000, learning_rate=0.1, seed=0, batch_size=1, optimizer='sgd', momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01, loss='mse')`: treino reproduzível por SGD, momentum ou AdamW com mini-batches. Defaults preservam o SGD por amostra. Atualização apenas após sucesso **e validação final de exportabilidade float32**; retorna as mesmas chaves `initialLoss`, `finalLoss` e `epochs`. Falha preserva pesos e referências internas originais.
- `training.train.train(..., validation_dataset=None, patience=None, min_delta=0.0, record_history=False)`: parâmetros opcionais P11 para seleção do checkpoint por validação, parada antecipada e registro por época; sem validação os valores retornados continuam iguais aos da API antiga (exceto com `record_history=True`). Conjuntos com entradas repetidas entre treino e validação são rejeitados.
- `training.train.dataset_loss(network, dataset, loss='mse')`: cálculo de perda sem gradientes. Também são suportadas `bce` (sigmoid, saída escalar) e `cce` (softmax, saída multiclasse); objetivos fora dessas combinações são rejeitados.
- `training.export.export_model(network)` e `save_model(network, filename)`: exportação JSON v1 float32. `save_model` grava primeiro um arquivo temporário no mesmo diretório, faz flush/fsync e o substitui com `os.replace`, removendo o temporário em caso de erro. Uma falha antes da substituição preserva o arquivo anterior; isso **não** garante durabilidade do diretório em queda de energia.

Os estados de momentum/AdamW existem apenas durante a chamada de treinamento, não são serializados nem retomados implicitamente. AdamW aplica weight decay aos pesos, não aos vieses. Consulte [treinamento avançado](training.md) para fórmulas, exemplos e limitações.

Não há treino automático no runtime JS nem garantia de convergência. O aplicativo consumidor cuida de normalizar vetores, carregar modelos, interpretar saídas e armazenar arquivos.

Compatibilidade **testada**: Node.js 22/24, Chrome headless e Manifest V3 Chrome for Testing (popup/worker). Não há validação específica de Workers Cloudflare, Firefox, Safari ou móveis.
