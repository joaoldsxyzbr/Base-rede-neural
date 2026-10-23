# Arquitetura inicial do Neural Core

**Estado:** v0.1 parcial: inferência JavaScript e treinamento supervisionado Python implementados. Testes de paridade cruzada (P3) e portabilidade em ambientes adicionais continuam pendentes.  
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

A estrutura JavaScript em `src/` e `tests/`, o treinamento em `training/`, os testes em `training_tests/` e o exemplo `examples/train_xor.py` estão implementados. `tests/parity.test.js` e fixtures de paridade permanecem planejados para P3.

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
4. **Integração:** exportar modelos e executar testes de paridade nas duas linguagens.
5. **Verificação:** rodar testes, benchmark local e exemplo genérico; revisar documentação e segurança.
6. **Expansão futura:** evoluir arquiteturas e otimizações apenas diante de necessidades e evidências; aplicações seguem desacopladas.

**Checkpoint P2:** 7 testes Python aprovados localmente em Python 3.13; XOR treinado com seed fixa reduziu a perda média de aproximadamente 0,14069 para 0,000143 (sobre as 4 amostras de treino). CI testa Python 3.11/3.13 e Node.js 22/24. Paridade numérica Python/JavaScript (P3), generalização e execução em navegador/extensão/Workers ainda não comprovadas. Nenhum projeto consumidor foi alterado.
