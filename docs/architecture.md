# Arquitetura inicial do Neural Core

**Estado:** especificação da v0.1; ainda não implementada.  
**Decisão:** começar com uma MLP pequena e comprovadamente correta, não com IA generativa.

## Objetivo e fronteiras

O Neural Core é uma biblioteca de código e formato de modelos. Não conhece PWA, extensão, banco de dados, Cloudflare, controles de ponto, finanças ou qualquer outra aplicação. Cada aplicação transforma suas entradas em vetores numéricos e interpreta a saída. Adaptadores e modelos específicos ficam **fora** do núcleo.

### Dentro da v0.1

- MLP feed-forward, camadas densas parametrizadas.
- Operações em vetores `Float32Array`, com validação de dimensões e valores finitos.
- Ativações necessárias para testes de classificação/regressão (começar com `tanh`, `sigmoid` e `linear`; ampliar somente mediante necessidade comprovada).
- Treinamento supervisionado por retropropagação e descida de gradiente, com inicialização determinística a partir de `seed` opcional.
- Inferência sem efeitos colaterais.
- Serialização e carregamento de arquitetura e pesos, com `formatVersion` e validações de tamanho, forma e valores.
- Testes nativos com Node.js; executar no navegador sem importar módulos exclusivos de Node.js.

### Fora da v0.1

- Transformer, chat generativo, tokenização, treinamento distribuído e GPU/WebGPU.
- Dependências, serviços pagos, telemetria, servidores ou inferência remota obrigatória.
- Persistência em um banco específico, interface gráfica e adaptadores de aplicação.
- Treinamento contínuo automático em produção.

## Estrutura prevista

```text
Neural-Core/
├── README.md
├── docs/
│   └── architecture.md
├── src/
│   ├── network.js       # topologia, pesos e inferência
│   ├── training.js      # cálculo de erro, gradientes e otimização
│   ├── model.js         # formato versionado e validação
│   └── index.js         # API pública, sem dependência de ambiente
├── tests/
│   ├── network.test.js  # dimensões, determinismo e resultados
│   ├── training.test.js # gradientes e aprendizado
│   └── model.test.js    # exportação, importação e falhas
├── examples/
│   └── xor.mjs          # exemplo didático genérico
└── package.json         # ESM e node --test, sem dependências
```

Os caminhos são propostos; nenhum arquivo de implementação foi criado.

## API pública inicial (proposta)

```javascript
import {
  createNetwork,
  predict,
  train,
  serializeModel,
  loadModel
} from "./src/index.js";

const network = createNetwork({
  layers: [2, 4, 1],
  hiddenActivation: "tanh",
  outputActivation: "sigmoid",
  seed: 42
});

const before = predict(network, [0, 1]);

const training = train(network, [
  { input: [0, 0], target: [0] },
  { input: [0, 1], target: [1] },
  { input: [1, 0], target: [1] },
  { input: [1, 1], target: [0] }
], { epochs: 2000, learningRate: 0.1 });

const stored = serializeModel(network);
const restored = loadModel(stored);
const after = predict(restored, [0, 1]);
```

A assinatura é preliminar. Não pressupor que os hiperparâmetros desse exemplo foram validados.

## Política de portabilidade

- JavaScript ESM, sem framework. `Float32Array` e laços nativos na primeira versão.
- O mesmo núcleo deve importar em navegadores, extensões e Node.js sem adaptações.
- Treinamento pode ocorrer fora do aplicativo; produção carrega somente pesos treinados.
- Workers gratuitos só receberão modelos **medidos** dentro dos limites reais de memória e CPU; compatibilidade não será presumida.
- Python, se necessário no futuro, fica restrito a ferramentas externas de treinamento/conversão, nunca como requisito de inferência.

## Validação e critérios de aceitação

1. **Matemática:** conferir gradientes com diferenças finitas e tolerâncias explícitas para ponto flutuante.
2. **Aprendizado:** comprovar redução de erro em dados sintéticos reproduzíveis; avaliar dados separados quando aplicável; comparar com baseline tradicional.
3. **Determinismo:** mesma configuração, amostras e seed devem produzir o mesmo resultado no mesmo ambiente.
4. **Persistência:** exportar e importar não pode alterar previsões além da tolerância definida.
5. **Segurança:** rejeitar dimensões incorretas, NaN, infinito, arquivos malformados, versões incompatíveis e tamanhos abusivos.
6. **Compatibilidade:** testar Node e navegador; verificar extensões e Workers separadamente, sem declarar compatibilidade antes de medir.
7. **Simplicidade:** manter uma única implementação, zero dependências de terceiros e nenhuma lógica específica de projetos.

## Etapas de implementação (ainda não executadas)

1. **Fundação:** `package.json`, API ESM e testes mínimos para validação de parâmetros.
2. **Inferência:** MLP com pesos e ativações; testar resultados e casos inválidos.
3. **Treinamento:** primeiro testes de gradiente, depois retropropagação/SGD e treinamento reproduzível.
4. **Persistência:** formato seguro e versionado, testes de ida e volta e rejeição de modelos inválidos.
5. **Verificação:** rodar `node --test`, exemplo XOR, benchmark simples, comparação com baseline e teste de importação em navegador.
6. **Expansão:** somente depois de evidência, incluir outras arquiteturas, otimizadores ou adaptadores como projetos separados.

Nenhum marco será considerado concluído antes dos testes reais. Nenhuma aplicação existente deve ser modificada nesta etapa.
