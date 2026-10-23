# Neural Core

Base neural **genérica, independente e reutilizável**, sem acoplamento a aplicações ou plataformas. Os projetos se adaptam ao Neural Core, e não o contrário.

## Linguagens e responsabilidades

- **JavaScript puro (ES Modules):** núcleo de **inferência** portátil para navegadores, extensões, Node.js e outros ambientes compatíveis; sem dependências externas obrigatórias.
- **Python:** ferramentas separadas de **treinamento, experimentação, avaliação e exportação** dos modelos. Não é exigido em tempo de execução nos aplicativos.
- **Formato compartilhado:** arquitetura, ativações e pesos exportados em formato versionado e validado; previsões JavaScript devem ser comparadas com referências produzidas pelo treino Python.

A primeira versão (v0.1) terá uma MLP pequena para classificação e regressão, **não** um modelo de conversação generativa. Treinamento e inferência serão separados; não haverá treinamento automático em produção.

## 7 regras fundamentais

1. **Simplicidade, robustez e qualidade:** código claro, seguro e fácil de manter.
2. **Universalidade e independência:** nenhuma lógica específica de projeto no núcleo.
3. **Arquitetura modular e extensível:** componentes com responsabilidades bem definidas.
4. **Portabilidade e compatibilidade:** JavaScript para execução; Python para ferramentas de treinamento.
5. **Desempenho e custo zero:** modelos compactos, processamento eficiente, sem servidor ou API paga obrigatórios.
6. **Aprendizado real e qualidade verificável:** testes reproduzíveis, métricas e comparação com alternativas tradicionais.
7. **Segurança, privacidade e confiabilidade:** processamento local, validação rigorosa, sem telemetria e sem comprometer compatibilidade.

## Uso atual

```javascript
import { loadModel, predict } from './src/index.js';
const model = loadModel({ formatVersion: 1, layers: [{ inputSize: 2, outputSize: 1, activation: 'linear', weights: [2, 3], biases: [1] }] });
const output = predict(model, [4, 5]); // Float32Array([24])
```

Execute `npm test` para os testes de inferência, ou `node examples/linear.mjs` para o exemplo de pesos manuais. O formato JSON usa `formatVersion: 1`, camadas densas e pesos por neurônio de saída.

## Treinamento Python (P2)

Treinamento supervisionado de MLP com retropropagação, SGD reproduzível por semente, validação e exportação de pesos para o mesmo formato JSON. Apenas a biblioteca padrão do Python é utilizada. O Python **não é necessário para executar os modelos** nos aplicativos.

```bash
python -m unittest discover -s training_tests -v
python -m examples.train_xor
```

O XOR é um teste controlado do aprendizado, não uma avaliação de generalização para aplicações reais. Para treinar/exportar programaticamente:

```python
from training.network import create_network
from training.train import train
from training.export import save_model

model = create_network([2, 4, 1], ["tanh", "sigmoid"], seed=7)
samples = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
metrics = train(model, samples, epochs=3500, learning_rate=0.35, seed=21)
save_model(model, "modelo.json")
```

Os testes do CI executam JavaScript em Node 22/24 e Python em 3.11/3.13. A comparação das previsões entre Python e JS permanece para o P3.

## Estado

**Inferência JS e treinamento Python implementados.** Testes locais de gradientes, aprendizagem XOR e exportação aprovados; paridade numérica Python × JavaScript (P3) e validação real de navegador/extensão/Worker ainda pendentes.

Consulte [a arquitetura, o formato dos modelos e as etapas de desenvolvimento](docs/architecture.md).
