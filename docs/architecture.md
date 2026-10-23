# Arquitetura inicial do Neural Core

**Estado:** v0.1 parcial: inferência JavaScript e treinamento supervisionado Python implementados. Testes de paridade cruzada (P3) implementados; P4 inclui benchmark Node e teste Chrome headless; P5 acrescenta validação em extensão MV3 isolada, com popup e service worker; Workers e mobile continuam sem validação específica.  
**Decisão:** desenvolver uma MLP pequena e matematicamente verificável. Não começar com IA generativa.

## 1. Independência

O Neural Core não conhece PWA, extensões, bancos de dados, Workers nem regras de negócio. Recebe vetores numéricos, executa inferência e devolve vetores. Modelos especializados, extração de atributos e interpretação dos resultados ficam **nos projetos consumidores**, fora do núcleo.

## 2. Linguagens: cada uma no seu ponto forte

| Responsabilidade | Linguagem | Ambiente |
| --- | --- | --- |
| Criação e treinamento de pesos | Python | Desenvolvimento / CI |
| Experimentos, avaliação e métricas | Python | Desenvolvimento / CI |
| Exportação de modelos portáteis | Python | Desenvolvimento / CI |
| Carregamento e validação dos modelos | JavaScript | Qualquer runtime JS compatível |
| Inferência leve e determinística | JavaScript | Navegadores, extensões e Node.js |

- **Python não é requisito de execução** para nenhum projeto consumidor.
- **JavaScript não precisa reimplementar a rotina de treinamento** na v0.1: teremos um treinamento canônico em Python e um único motor de inferência em JS.
- Começar com a **biblioteca padrão do Python** e recursos nativos do JavaScript. Adicionar apenas ferramentas oficiais/estabelecidas quando os ganhos forem demonstrados, preservando custo zero e a portabilidade.
- Nenhuma API de IA, serviço pago, telemetria, infraestrutura de servidor ou armazenamento específico é necessário.

## 3. Escopo da v0.1

### Incluído

- MLP feed-forward com camadas densas e dimensões configuráveis.
- Ativações `tanh`, `sigmoid` e `linear` como conjunto inicial mínimo.
- Treinamento supervisionado em Python por retropropagação e SGD, com semente reproduzível.
- Exportação de arquitetura, pesos e vieses para arquivo de modelo versionado.
- Inferência JavaScript por `Float32Array` e validação de entrada e modelo.
- Testes de gradiente, aprendizado real, persistência e equivalência numérica entre Python e JS.
- Documentação simples de uso, limites e resultados medidos.

### Excluído

- Transformers, chat generativo, tokenização, WebGPU e treinamento distribuído.
- Treinamento contínuo automático nos aplicativos.
- Adaptadores para aplicações concretas, interface gráfica e persistência em banco de dados.
- Dependência de Cloudflare, Node.js ou browser específica dentro do núcleo de inferência.

## 4. Estrutura proposta

```text
Neural-Core/
├── README.md
├── docs/
│   └── architecture.md
├── src/
│   ├── network.js           # carregar pesos e executar inferência
│   ├── model.js             # serialização/validação do formato
│   └── index.js             # API pública JS
├── training/
│   ├── network.py           # MLP de referência e retropropagação
│   ├── train.py             # treino e avaliação reproduzíveis
│   └── export.py            # exportação do formato compartilhado
├── tests/
│   ├── network.test.js      # inferência e validação
│   ├── model.test.js        # formato e entradas malformadas
│   ├── parity.test.js       # vetores de referência vindos do Python
│   └── fixtures/            # modelos mínimos e resultados esperados
├── training_tests/
│   ├── test_gradients.py
│   └── test_learning.py
├── examples/
│   └── xor.mjs              # exemplo genérico
└── package.json             # ESM, testes nativos, sem dependências
```

A estrutura JavaScript em `src/` e `tests/`, o treinamento em `training/`, os testes em `training_tests/` e o exemplo `examples/train_xor.py` estão implementados. `tests/parity.test.js`, `tests/fixtures/parity.json`, `training/parity_fixture.py` e `training_tests/test_parity.py` foram adicionados no P3.

## 5. Contrato comum dos modelos

Formato inicial planejado: JSON independente de linguagem, com `formatVersion: 1`, lista ordenada de camadas, dimensões, nome da ativação, pesos achatados em **ordem de linhas** (`peso[neurônio_de_saida][neurônio_de_entrada]`) e vieses por neurônio de saída. Valores numéricos devem ser finitos e convertidos para precisão `float32` na exportação. Cada camada deve declarar explicitamente as dimensões esperadas.

O carregador JS deve rejeitar versões desconhecidas, topologias inválidas, incompatibilidade de dimensões, valores não finitos, tamanhos abusivos e ativações não suportadas. O aplicativo é responsável pela obtenção e pelo controle de acesso ao arquivo; carregar um modelo não deve executar código arbitrário.

A serialização é um **formato de intercâmbio**, não uma dependência do Python. Poderemos mudar seu armazenamento interno sem quebrar a API pública, desde que o contrato versionado seja respeitado.

### API pública JS (implementada)

```javascript
import { loadModel, predict } from "./src/index.js";

const model = loadModel(exportedModel);
const output = predict(model, [0.2, 0.8]);
```

`exportedModel` representa os dados importados do arquivo; `predict` não treina nem modifica os pesos. As assinaturas serão testadas antes de serem consideradas estáveis.

## 6. Portabilidade e desempenho

O motor de inferência será um módulo ES padrão sem imports exclusivos de Node.js, DOM, APIs de rede ou Cloudflare. A execução deve ocorrer localmente sempre que o ambiente suportar o tamanho do modelo.

**Cloudflare Workers gratuito não implica compatibilidade automática.** Publicaremos nesse ambiente somente modelos e cargas que comprovadamente respeitem limites reais de CPU, memória e tamanho.

## 7. Critérios de aceitação

1. **Gradientes corretos:** comparar gradientes analíticos Python com diferenças finitas em exemplos controlados.
2. **Treinamento real:** demonstrar redução do erro em dados sintéticos reproduzíveis (XOR como diagnóstico); medir também generalização em dados separados quando adequado.
3. **Comparação:** confrontar resultado com uma solução tradicional simples, sem afirmar superioridade apenas por usar rede neural.
4. **Paridade:** comparar previsões Python e JavaScript no mesmo modelo exportado com tolerância numérica documentada.
5. **Persistência:** exportar, importar e prever sem mudanças indevidas no resultado.
6. **Robustez:** testar entradas inválidas, NaN, infinito, modelos adulterados, formato desconhecido e dimensões incompatíveis.
7. **Portabilidade:** executar em Node.js e navegador; testar extensões e Workers à parte antes de anunciar suporte.
8. **Leveza:** medir tempo e consumo de memória para tamanhos de modelo representativos; documentar limites.
9. **Sem dependências externas obrigatórias:** `node --test` e `python -m unittest discover -s training_tests`.

## 8. Implementação prevista (não executada)

1. **Contrato:** definir JSON versionado, fixtures e testes de entradas inválidas.
2. **Inferência JS:** implementar camadas densas, ativações e testes matemáticos.
3. **Treinamento Python (P2 concluído):** MLP, retropropagação, SGD reproduzível, exportação float32, testes numéricos de gradientes e aprendizagem XOR.
4. **Integração (P3 concluído):** gerar referências em Python após treinamento, exportar float32, comparar valores no JavaScript e verificar serialização nos casos reproduzíveis.
5. **Verificação (P4):** benchmark Node para três tamanhos e teste de inferência no Chrome real com fixture P3; métricas informativas e sem thresholds frágeis.
6. **Expansão futura:** evoluir arquiteturas e otimizações apenas diante de necessidades e evidências; aplicações seguem desacopladas.

**Checkpoint P3:** treino e exportação Python confrontados com inferência JS nos mesmos modelos (XOR treinado, redes lineares de múltiplas saídas, combinações de ativações e sigmoid com entradas extremas). A tolerância de saída é `1e-5 + 1e-5 × abs(referência)` devido ao uso de float32 no JS. A fixture fica versionada para permitir `npm test` independente de Python; um job de integração no CI gera uma fixture fresca em Python para evitar regressões mascaradas por snapshots antigos. Generalização, benchmarks e execução em navegadores/extensões/Workers ainda não foram comprovados. Nenhum projeto consumidor foi modificado.

## P4 — Benchmark e teste real de navegador

- `benchmarks/inference.mjs` informa latência mediana de cinco lotes (200 execuções de aquecimento) em 3 topologias, tempo de importação JSON, tamanho dos parâmetros em `float32` e variação de memória observada. Não há limiar rígido de performance na CI compartilhada.
- `tests/browser-smoke.html` importa `src/index.js` diretamente do navegador, carrega a fixture Python/JS e compara as previsões dentro da tolerância numérica do P3. A importação e persistência são verificadas no navegador.
- `scripts/browser-smoke.mjs` disponibiliza somente cinco arquivos locais num servidor HTTP em `127.0.0.1` e abre o Chrome/Chromium headless, sem bibliotecas extras, para conferir o resultado no DOM.
- O CI adiciona jobs de **Chrome headless** e **benchmark Node 24**, preservando os testes JS/Python/paridade anteriores.
- **Limites:** benchmarks em runners compartilhados não são medição definitiva de hardware real, não representam memória de uma extensão/PWA e não comprovam execução em Cloudflare Workers. Uma extensão MV3 deve receber uma verificação específica de suas políticas e lifecycle; Workers precisam de uma medição no seu ambiente real.

Detalhes em [performance.md](performance.md).

## P5 — Extensão independente Manifest V3

- O P5 mantém o Neural Core genérico e o empacota somente no **teste** `tests/extension-mv3/`, sem incluir lógica para um aplicativo específico no núcleo.
- O manifest MV3 não solicita permissões de sites, armazenamento, rede externa ou telemetria.
- O `popup.js` compara os 19 vetores da referência Python × JS, testa carga e serialização dos modelos e verifica que está em `chrome-extension://`.
- O `worker.js` é um service worker `type: module`, importa os mesmos arquivos `src/*.js`, registra o listener síncrono e responde a uma mensagem do popup após executar `predict`.
- O runner `scripts/extension-smoke.mjs` copia temporariamente o código, carrega a extensão unpacked no Chromium/Chrome for Testing e **confere o resultado no DOM**; não confunde teste de página HTML comum com execução em contexto de extensão.
- O CI testa a extensão num Chrome for Testing oficial instalado em diretório temporário, sem bibliotecas npm e sem exigir instalação nos projetos consumidores.
- **Limites:** o teste não comprova funcionamento dentro da extensão de modo escuro, lifecycle completo da MV3, mobile, Chrome Web Store ou Cloudflare Workers.

Detalhes em [extension-mv3.md](extension-mv3.md).

## P6 — Robustez do treinamento

Treinamento Python com commit seguro: `train` valida o modelo usando `export_model`, valida todo o dataset, treina uma cópia local dos pesos e vieses, verifica a perda final e somente então atualiza as listas originais. Em caso de divergência ou exceção durante o cálculo, o objeto original permanece sem mudanças. Treinamento bem-sucedido continua atualizando as listas **in-place** para compatibilidade. A cópia tem limite de 250.000 parâmetros do contrato v1.

A função `dataset_loss` calcula previsão e erro sem a retropropagação. Entradas Python com inteiros grandes demais para conversão finita geram `ValueError` consistente. A exportação rejeita `formatVersion: true` e parâmetros fora do intervalo float32. O runtime JavaScript não mudou; P7 (generalização e comparação de métodos) e P8 (governança, ambiente CI reproduzível) continuam fora do P6.

## P7 — Evidência de generalização e comparação

Foi incluída uma bancada de avaliação **offline**, independente do núcleo, em `evaluation/benchmark.py`. Ela usa conjuntos de treino e teste independentes para duas tarefas sintéticas; calcula acurácia, Brier, erro quadrático médio (MSE), erro absoluto médio (MAE), médias e dispersão em várias sementes. Os baselines são classificador linear, maioria, mínimos quadrados e média constante.

Os experimentos distinguem uma relação genuinamente não linear, em que a MLP pode ter vantagem, de um problema linear, em que um método tradicional é suficiente. O objetivo é verificar aprendizado e generalização **apenas nas distribuições artificiais documentadas**, não alegar superioridade sobre técnicas clássicas em geral. A inferência JavaScript e o treinamento já existentes não sofreram mudança.

Ver [docs/evaluation.md](evaluation.md). P8 (documentação/API/licenciamento e CI reprodutível) permanece como etapa separada.
