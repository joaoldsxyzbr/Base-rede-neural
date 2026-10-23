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

Os testes do CI executam JavaScript em Node 22/24 e Python em 3.11/3.13.

## Paridade Python × JavaScript (P3)

Um gerador Python produz pesos exportados em float32 e previsões de referência com casos lineares, redes multicamadas, sigmoid extrema e uma MLP treinada com XOR. O teste Node carrega **esses mesmos pesos** e verifica as saídas com tolerância `1e-5 + 1e-5 × abs(referência)`, inclusive após serializar e carregar novamente o modelo.

```bash
python -m training.parity_fixture > tests/fixtures/parity.json
node --test tests/parity.test.js
```

O CI inclui também um job separado que gera novamente o conjunto de referência em Python e o utiliza diretamente nos testes JavaScript, impedindo que um arquivo antigo de exemplo esconda regressões. A fixture versionada permite executar `npm test` sem instalar Python. Isso valida paridade numérica nos casos testados, não precisão ou generalização em aplicações reais.

## Desempenho e portabilidade (P4)

Os benchmarks usam modelos genéricos de três tamanhos, aquecimento e mediana de cinco rodadas. Medem tempo de carregamento, latência por previsão, tamanho de pesos e observações de memória no processo Node.js.

```bash
node benchmarks/inference.mjs
```

O smoke test usa **um Chrome/Chromium real em modo headless**, HTTP apenas em `127.0.0.1`, módulos JavaScript nativos e modelos de referência do P3. Sem serviços externos ou dependências npm:

```bash
node scripts/browser-smoke.mjs
```

Para o teste de navegador, é necessário ter Chrome/Chromium instalado (ou informar `CHROME_BIN` com o caminho do executável). O CI usa a instalação de Chrome do runner Ubuntu. Consulte [metodologia e limites dos benchmarks](docs/performance.md). A execução como extensão Manifest V3 e no Cloudflare Workers continua exigindo testes próprios: compatibilidade no Chrome **não prova** esses cenários.

## Extensão independente Manifest V3 (P5)

O teste em `tests/extension-mv3/` é uma extensão MV3 **exclusivamente para validação**. Não está integrada ao Dark Reader nem a nenhum aplicativo. Um script empacota temporariamente o núcleo JS original e a fixture de referência dentro da extensão. Não instala bibliotecas externas, não coleta dados, não tem permissões de sites, nem solicita acesso a abas.

O popup verifica os **19 vetores da paridade Python × JS**. O service worker MV3, carregado como módulo ES, importa o mesmo núcleo para executar uma previsão de referência (`[4,5] → [24]`). O teste falha se a extensão não carregar ou se qualquer contexto não responder corretamente.

```bash
node --test tests/extension-mv3.test.js
node scripts/extension-smoke.mjs
```

Para o smoke real, use **Chromium** ou **Chrome for Testing** (caminho em `CHROME_BIN`), que permitem carregar extensões temporárias no ambiente de testes. O CI instala o Chrome for Testing a partir do serviço oficial do Google, apenas no job de teste. Nenhuma instalação é necessária para os usuários finais do motor. Consulte [documentação do P5](docs/extension-mv3.md).

## Estado

**Inferência JS, treinamento Python e testes de paridade (P3) implementados.** O CI valida os casos de referência entre linguagens; benchmarks Node e smoke test em navegador adicionados ao CI (P4); P5 acrescenta um teste independente de extensão Manifest V3; Workers e dispositivos móveis continuam sem validação específica.

Consulte [a arquitetura, o formato dos modelos e as etapas de desenvolvimento](docs/architecture.md).

## P6 — Robustez e validação numérica

- O treinamento Python agora **preserva integralmente o modelo original se ocorrer erro**, inclusive divergência numérica. Os pesos são calculados em uma cópia temporária e copiados para os arrays existentes somente após treino e avaliação bem-sucedidos.
- A avaliação de perda (`dataset_loss`) usa apenas inferência, evitando retropropagação desnecessária.
- Valores numéricos extremos e versões de formato inválidas são rejeitados com `ValueError` controlado, sem aceitar booleanos como versão.
- Novos testes em `training_tests/test_robustness.py` cobrem rollback, atualização in-place, números extremos e ausência de cálculo de gradientes na avaliação.

Execute `python -m unittest discover -s training_tests -v`. Esse isolamento custa uma cópia dos pesos **apenas durante o treinamento offline**; o runtime JS de inferência não foi alterado. Não há bibliotecas adicionais.

## P7 — Avaliação científica reproduzível

O módulo offline `evaluation/benchmark.py` avalia duas tarefas **sintéticas** com conjuntos de treino e teste independentes, sementes explícitas e baselines simples. Não afeta a API pública nem o desempenho da inferência JavaScript.

- **Classificação não linear:** MLP 2→8→1 versus classificador linear sigmoide e frequência da classe majoritária. Mede acurácia e Brier no treino, no conjunto de teste sem ruído e no mesmo conjunto de teste com ruído adicional.
- **Regressão linear:** camada linear 1→1 versus mínimos quadrados analíticos e predição da média. Mede MSE e MAE no treino e no teste.
- **Reprodução:** sementes 7, 19 e 31, 96 amostras para treinamento e 128 para teste em cada repetição; reporta médias e desvio-padrão populacional. Não consulta dados de teste para treinar ou selecionar checkpoint.

```bash
python -m unittest discover -s training_tests -v
python -m evaluation.benchmark
```

O CI executa os testes científicos em Python 3.11 e 3.13 e imprime o relatório no job dedicado. Metodologia, resultados locais e limitações: [docs/evaluation.md](docs/evaluation.md). Nenhuma biblioteca adicional é necessária.
