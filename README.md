# Neural Core

Base neural **genérica, simples, portátil e independente**, com treinamento offline em Python e inferência local em JavaScript puro. Versão inicial `0.1.0`.

- **JavaScript ES Modules:** carregar/validar, prever e serializar modelos de redes densas pequenas, sem dependência de DOM, Node.js, Cloudflare ou servidor.
- **Python (stdlib):** treinamento real (SGD + retropropagação), exportação de modelos JSON, experimentos, métricas e comparação com métodos tradicionais.
- **Privacidade e custo zero:** nenhuma telemetria, assinatura ou biblioteca externa obrigatória.
- **Limites:** não é um LLM, não há treinamento contínuo em produção e o XOR/benchmarks sintéticos não comprovam resultados em dados reais.

## Uso JavaScript

```javascript
import { loadModel, predict, serializeModel } from './src/index.js';

const model = loadModel({
  formatVersion: 1,
  layers: [{ inputSize: 2, outputSize: 1, activation: 'linear',
             weights: [2, 3], biases: [1] }]
});
const result = predict(model, [4, 5]); // Float32Array([24])
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

O treinamento é isolado: se falhar, os pesos originais não são alterados. O formato exportado é carregável em JavaScript.

## Verificações reproduzíveis

```bash
npm test
python -m unittest discover -s training_tests -v
python -m evaluation.benchmark
node benchmarks/inference.mjs
```

O GitHub Actions executa esses testes, verifica paridade Python×JS e inferência no Chrome headless e em uma extensão MV3 (popup e service worker). Chrome é ferramenta de CI, **não** dependência do núcleo. Dados de avaliação são sintéticos.

## Documentação

- [Contrato da API e formato JSON](docs/api.md)
- [Arquitetura atual](docs/architecture.md)
- [Experimentos e baselines](docs/evaluation.md)
- [Benchmarks](docs/performance.md)
- [Teste MV3](docs/extension-mv3.md)
- [Roadmap](ROADMAP.md)
- [Auditoria final e limitações conhecidas](docs/audit.md)
- [Segurança](SECURITY.md)

Princípios: simplicidade, universalidade, modularidade, portabilidade, desempenho e custo zero, aprendizado verificável, segurança e privacidade. Aplicativos se adaptam ao núcleo, nunca o contrário.

**Licença:** MIT — [LICENSE](LICENSE). Dados/modelos de terceiros mantêm seus direitos.
