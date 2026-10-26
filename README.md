# Neural Core

Base neural **genérica, simples, portátil e independente**, com treinamento offline em Python e inferência local em JavaScript puro. Versão inicial `0.1.0`.

- **JavaScript ES Modules:** carregar/validar, prever e serializar modelos de redes densas pequenas (linear, tanh, sigmoid, ReLU e softmax), sem dependência de DOM, Node.js, Cloudflare ou servidor.
- **Python (stdlib):** treinamento real (SGD, momentum, AdamW, mini-batches + retropropagação), exportação de modelos JSON, experimentos sintéticos e avaliação em dados reais de classificação (Iris) e regressão (Airfoil) com comparação a métodos tradicionais.
- **Privacidade e custo zero:** nenhuma telemetria, assinatura ou biblioteca externa obrigatória.
- **Limites:** não é um LLM, não há treinamento contínuo em produção e o XOR/benchmarks sintéticos não comprovam resultados em dados reais.

## Uso JavaScript

```javascript
import { loadModel, predict, predictBatch, serializeModel, getCapabilities, describeModel } from './src/index.js';

const model = loadModel({
  formatVersion: 1,
  layers: [{ inputSize: 2, outputSize: 1, activation: 'linear',
             weights: [2, 3], biases: [1] }]
});
const result = predict(model, [4, 5]); // Float32Array([24])
const predictions = predictBatch(model, [[4, 5], [1, 2]]); // duas saídas independentes
const features = getCapabilities(); // formatos e ativações realmente disponíveis
const info = describeModel(model); // dimensões e custo, sem copiar pesos
const json = JSON.stringify(serializeModel(model));
```

Em Node.js, o nome próprio do pacote também funciona dentro do repositório, como `import { predict } from 'neural-core'`. O pacote permanece **privado e não publicado no npm**. Em navegadores e extensões, importe `src/index.js` por caminho relativo com seus módulos. Obter/guardar modelos, extrair atributos e interpretar resultados são responsabilidades do consumidor.

## Treinar offline

```python
from training.network import create_network
from training.train import train
from training.export import save_model

model = create_network([2, 4, 1], ['tanh', 'sigmoid'], seed=7)
samples = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
metrics = train(model, samples, epochs=3500, learning_rate=0.35, seed=21)
save_model(model, 'modelo.json')
```

O treinamento é isolado: se falhar ou gerar pesos fora do intervalo float32, os pesos originais não são alterados. SGD e MSE continuam sendo os padrões para compatibilidade. `save_model` substitui arquivos de forma atômica no mesmo diretório (conforme suporte do sistema de arquivos). O formato exportado é carregável em JavaScript.

## Verificações reproduzíveis

```bash
npm test
python -m unittest discover -s training_tests -v
python -m evaluation.benchmark
python -m evaluation.quality
python -m evaluation.real_data
python -m evaluation.airfoil
node benchmarks/inference.mjs
node benchmarks/batch.mjs
node --expose-gc benchmarks/stress.mjs
python -m training.interop_fixture
```

O GitHub Actions executa os testes e a avaliação científica, verifica paridade Python×JS, Chrome, extensão MV3, Firefox/WebKit em Linux e o runtime Cloudflare `workerd` local. Playwright/Wrangler são ferramentas **somente de CI**, instaladas por `npm ci --prefix ci-tools` com lockfile independente; não são dependências do núcleo. WebKit emulado não comprova Safari/iOS físico; testes científicos usam dados sintéticos e bases reais Iris e Airfoil, que não comprovam desempenho geral em aplicações de produção.

## Documentação

- [Contrato da API e formato JSON](docs/api.md)
- [Treinamento avançado P10](docs/training.md)
- [Política de compatibilidade e versionamento](docs/compatibility.md)
- [Contrato interoperável e evolução P14](docs/interoperability.md)
- [Arquitetura atual](docs/architecture.md)
- [Experimentos, validação P11 e baselines](docs/evaluation.md)
- [Benchmarks](docs/performance.md)
- [Matriz de portabilidade P13](docs/portability.md)
- [Teste MV3](docs/extension-mv3.md)
- [Roadmap](ROADMAP.md)
- [Auditoria de consolidação P15](docs/audit-p15.md)
- [Auditoria histórica P8](docs/audit.md)
- [Proteção da branch main (configuração administrativa)](docs/branch-protection.md)
- [Segurança](SECURITY.md)

Princípios: simplicidade, universalidade, modularidade, portabilidade, desempenho e custo zero, aprendizado verificável, segurança e privacidade. Aplicativos se adaptam ao núcleo, nunca o contrário.

**Licença:** MIT — [LICENSE](LICENSE). Dados/modelos de terceiros mantêm seus direitos.
