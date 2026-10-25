# Arquitetura do Neural Core v0.1.0

## Objetivo

Uma base neural pequena, genérica e reutilizável. **Python treina offline; JavaScript faz inferência local.** Nenhum projeto, aplicação, backend, extensão, navegador ou banco de dados é obrigatório.

## Organização implementada

| Área | Responsabilidade |
| --- | --- |
| `src/` | API ES Modules, contrato de capacidades, descrição de modelos, validação, inferência float32 e lote com buffers temporários |
| `training/` | MLP, gradientes, SGD/momentum/AdamW, mini-batches, classificação e exportação em Python padrão |
| `evaluation/` | Experimentos e comparação com baselines tradicionais |
| `tests/` | Testes nativos Node, paridade, navegador e Manifest V3 |
| `training_tests/` | Testes Python matemáticos, numéricos, científicos e de ferramentas |
| `benchmarks/` | Medição informativa de desempenho e memória |
| `scripts/` | Ferramentas de teste/CI; não integram o runtime |
| `docs/` | Contratos e evidências |

O contrato `formatVersion: 1` aceita MLP densa, ativações `linear`, `tanh`, `sigmoid`, `relu` e `softmax` (apenas na saída), pesos em `[saída][entrada]` e vieses. O treinamento usa números de precisão dupla Python e exporta pesos float32; o JavaScript usa `Float32Array`. Confira os detalhes em [api.md](api.md).

Os parâmetros internos não são modificáveis por consumidores via `model.layers` (que retorna cópias). O treinamento Python preserva os pesos originais caso uma execução falhe.

## Evidências

- **P0–P3:** CI, testes matemáticos/backprop, aprendizado XOR, persistência e paridade Python × JavaScript.
- **P4–P5:** benchmarks Node, smoke test real Chrome headless e extensão Manifest V3 isolada.
- **P6:** integridade do treinamento com commit seguro e validação numérica.
- **P7:** conjuntos artificiais de treino/teste separados, ruído e comparação com métodos tradicionais.
- **P8–P9:** contrato API, documentação, licenciamento, validação de ambiente, persistência segura e revisão.
- **P10:** mini-batches, momentum, AdamW, ReLU/He, BCE, softmax/CCE e paridade Python × JS.
- **P12:** inferência em lote, buffers temporários, paridade com inferência individual, benchmark e testes Chrome/MV3.
- **P13:** Firefox/WebKit Linux, emulação móvel e workerd local, sem alterar o motor.
- **P14:** declaração de capacidades e descritores Python/JS; teste de contrato gerado offline e validado no CI.
- **P15:** validação de campo reservado `operator`, testes gerativos e stress, lockfile das ferramentas de teste separado do runtime e documentação de proteção da main.

Resultados de generalização estão restritos às distribuições artificiais de [evaluation.md](evaluation.md); latência em runner é apenas referência e está documentada em [performance.md](performance.md).

## Segurança, portabilidade e não objetivos

O runtime não usa rede, telemetria, Node, DOM, Cloudflare, eval ou APIs pagas. A origem/armazenamento do modelo, extração de atributos e uso da previsão são do consumidor.

Não comprovados: Cloudflare Workers publicados, Safari nativo, dispositivos móveis físicos, lifecycle extensivo de service worker ou aplicação real. Não incluídos: LLM, transformers, WebGPU, autoaprendizado, integração com projetos específicos e publicação npm. Extensões futuras requerem testes e evidência concreta.
