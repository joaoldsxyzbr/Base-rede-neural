# Contrato interoperável e evolução de operadores — P14

O Neural Core é uma base **universal e independente de aplicações**. O objetivo P14 não é adicionar complexidade para suportar arquiteturas não solicitadas; é fornecer um contrato verificável que ajude consumidores a decidir se podem executar um modelo. O modelo `formatVersion: 1` permanece o único formato de modelo implementado.

## Consultar capacidades sem tentativa e erro

O JavaScript expõe `getCapabilities()` e o Python `training.export.get_capabilities()`. Os dois devolvem o mesmo objeto JSON:

```json
{
  "formatVersions": [1],
  "operators": ["dense"],
  "activations": ["linear", "relu", "sigmoid", "softmax", "tanh"],
  "limits": { "maxLayers": 16, "maxWidth": 4096, "maxParameters": 250000 }
}
```

- `dense`: único tipo de camada implementado. A ordem dos pesos é `[neuronioSaida][neuronioEntrada]`. Operadores não listados são rejeitados; não existe registro dinâmico por plugin.
- As ativações realmente disponíveis para inferência são listadas no código de execução JS e verificadas no contrato Python pelo CI.
- O JS retorna uma estrutura **congelada**; o Python retorna um novo `dict` independente. Não é possível alterar a compatibilidade da instância editando a resposta.
- Os três limites descritos são **compartilhados**. Limites adicionais específicos do runtime incluem tamanho máximo de string JSON na leitura JS e tamanho máximo de lote (4.096 amostras). Eles não são negociados pelo contrato comum.
- Capacidades anunciadas **não substituem** validação estrutural: em particular, `softmax` só é aceita na última camada e com pelo menos duas saídas.

## Descrever modelos sem copiar parâmetros

`describeModel(loaded)` e `training.export.describe_model(model)` retornam o mesmo formato:

```json
{
  "formatVersion": 1,
  "inputSize": 2,
  "outputSize": 2,
  "layerCount": 1,
  "parameterCount": 6,
  "layers": [
    {"inputSize": 2, "outputSize": 2, "activation": "softmax", "parameterCount": 6}
  ]
}
```

O JS requer um modelo criado com `loadModel`. Lê somente as camadas internas já validadas e **não copia os pesos**, não expõe buffers internos e não realiza inferência. Cada descritor é imutável. A implementação Python confirma primeiro que o modelo pode ser exportado para float32 e não altera o original. Descritores não incluem rótulos, normalização de entrada ou objetivos de negócio — esses itens pertencem ao projeto consumidor.

```javascript
import {loadModel, getCapabilities, describeModel, predict} from './src/index.js';

const capabilities = getCapabilities();
const source = {formatVersion:1, layers:[
  {inputSize:2, outputSize:1, activation:'linear', weights:[2,3],biases:[1]}
]};
if (!capabilities.formatVersions.includes(source.formatVersion))
  throw new Error('Formato incompatível');
const model = loadModel(source); // valida pesos, limites e operadores
const info = describeModel(model);
const output = predict(model, [4,5]);
```

## Conformidade automatizada Python × JavaScript

```bash
python -m training.interop_fixture > /tmp/neural-core-interop.json
NEURAL_CORE_INTEROP_FIXTURE=/tmp/neural-core-interop.json node --test tests/interop.test.js
```

O Python gera modelos determinísticos representativos (incluindo ReLU, softmax, tanh, sigmoid e linear) e descritores numéricos. O JS precisa concordar **exatamente** nos nomes de operadores, ativações, limites, descrições de camada, contagens de parâmetros e serialização normalizada. Testes unitários dos dois lados cobrem imutabilidade, entradas inválidas e versões desconhecidas.

O CI executa essa verificação no job existente de paridade, sem adicionar um novo serviço ou dependência. Fixtures e resultados não são persistidos em produção. O teste estabelece compatibilidade v1 dos exemplos, não a correção de qualquer arquitetura futura.

## Regras para extensões futuras

1. Adicionar uma nova ativação apenas após implementar **inferência JS, treinamento/gradiente Python, validação estrutural, exportação e testes numéricos**; atualizar a declaração de capacidades e a fixture. Evitar compatibilidade apenas nominal.
2. Um operador diferente de `dense` ou qualquer formato incompatível exige proposta explícita de `formatVersion: 2`, **sem alterar a interpretação de v1**. Não aceitar v2 automaticamente.
3. Se um conversor externo for justificado (ONNX etc.), mantê-lo em ferramenta **offline/opcional** e validar sua equivalência numérica. Não adicionar SDK, gateway ou parser de terceiros ao runtime universal.
4. Não acrescentar quantização, CNN, RNN, transformers ou representações binárias sem caso independente e benchmarks que mostrem benefício líquido.
5. Preservar `loadModel`, `predict`, `predictBatch`, `serializeModel` e as demais funções públicas; qualquer quebra precisa de nova versão de pacote documentada e testes de migração.

Este contrato serve a todos os projetos consumidores, mas nenhum deles determina a arquitetura do Neural Core.
