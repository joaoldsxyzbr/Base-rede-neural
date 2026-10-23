# Medições e portabilidade — P4

## Objetivo

Medir o custo da inferência MLP em JavaScript puro, sem adicionar dependências ou converter o Neural Core para uma aplicação específica. Medir e testar **não implica otimizar automaticamente** o código.

## Benchmark

Executar `node benchmarks/inference.mjs`. O comando imprime JSON legível com:

- `small` = 8→16→4 (212 parâmetros);
- `medium` = 32→64→16 (3.152 parâmetros);
- `large` = 128→128→32 (20.640 parâmetros);
- armazenamento de pesos `float32`, tamanho do JSON, tempo de carregamento, latência por predição (5 lotes e mediana, após 200 aquecimentos), e delta de memória residente/heap observado.

As topologias e entradas são artificiais e **não** representam casos reais. Os deltas de RSS e heap são medições grosseiras do processo inteiro, dependem de GC, JIT e alocador e podem ser zero ou negativos. `weightBytesFloat32` mede somente parâmetros e não é uso total de memória do modelo. Para comparar versões, medir no mesmo hardware/versão do Node, com várias rodadas. Resultados em runners compartilhados são orientativos; por isso **não** se usa um threshold frágil como critério de aprovação.

## Portabilidade em navegador

Executar `node scripts/browser-smoke.mjs` num host que tenha Chrome ou Chromium, ou passar `CHROME_BIN` com o executável. O script serve arquivos predefinidos em HTTP localhost, carrega a página do teste num navegador real e verifica dados no DOM. Não usa browser automation de terceiros, download automático, nem serviços externos.

A página importa a implementação **original** `src/index.js`, executa os casos de paridade gerados em Python, valida `loadModel`, `predict` e `serializeModel`, e compara saídas com `1e-5 + 1e-5 × |referência Python|`. Se faltar o navegador, o teste falha (não é silenciosamente pulado). O CI executa esse teste em Chrome headless e registra sucesso/falha.

## Limites do escopo P4

- Chrome headless no CI comprova execução nesse navegador, não em todos os navegadores ou versões.
- Não implica suporte validado a extensões Manifest V3; extensões exigem testes próprios, inclusive contexto de service worker e permissões.
- Não implica compatibilidade garantida com Workers Free; é necessário medir limites de CPU/memória nos Workers.
- Nenhum aplicativo consumidor é modificado pelo P4, e os testes não exigem APIs pagas ou Python no navegador.

## Referência inicial medida no GitHub Actions

Execução de **23/10/2026**, [workflow P4](https://github.com/joaoldsxyzbr/Base-rede-neural/actions/runs/40005582052), Node.js `24.21.0`, Linux x64, runner Ubuntu compartilhado.

| Topologia | Parâmetros | Pesos float32 (bytes) | Carregamento JSON (ms) | Inferência mediana (µs) |
| --- | ---: | ---: | ---: | ---: |
| 8→16→4 | 212 | 848 | 0,314 | 1,032 |
| 32→64→16 | 3.152 | 12.608 | 0,701 | 8,520 |
| 128→128→32 | 20.640 | 82.560 | 1,794 | 32,048 |

As medidas foram extraídas dos logs do job `Benchmark / Node 24`. **Não são garantias de desempenho** e não incluem o custo total de memória da aplicação. O job `Navegador / Chrome headless` passou com **19 previsões de referência**. A execução direta no container local não completou por limitações do processo Chromium no ambiente local; o teste foi confirmado no runner real do GitHub.

As próximas melhorias de otimização devem ser avaliadas com o mesmo benchmark, com diversas repetições em hardware controlado. Não associar o tempo de inferência a treinamento, precisão de modelos reais, extensões ou Workers sem medições adicionais.
