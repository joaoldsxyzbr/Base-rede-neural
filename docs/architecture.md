# Arquitetura do Neural Core v0.1.0

## Objetivo

Uma base neural pequena, genérica e reutilizável. **Python treina offline; JavaScript faz inferência local.** Nenhum projeto, aplicação, backend, extensão, navegador ou banco de dados é obrigatório.

## Organização implementada

| Área | Responsabilidade |
| --- | --- |
| `src/` | API ES Modules, validação de modelo e inferência float32 |
| `training/` | MLP, gradientes, SGD e exportação em Python padrão |
| `evaluation/` | Experimentos e comparação com baselines tradicionais |
| `tests/` | Testes nativos Node, paridade, navegador e Manifest V3 |
| `training_tests/` | Testes Python matemáticos, numéricos, científicos e de ferramentas |
| `benchmarks/` | Medição informativa de desempenho e memória |
| `scripts/` | Ferramentas de teste/CI; não integram o runtime |
| `docs/` | Contratos e evidências |

O contrato `formatVersion: 1` aceita MLP densa, ativações `linear`, `tanh` e `sigmoid`, pesos em `[saída][entrada]` e vieses. O treinamento usa números de precisão dupla Python e exporta pesos float32; o JavaScript usa `Float32Array`. Confira os detalhes em [api.md](api.md).

Os parâmetros internos não são modificáveis por consumidores via `model.layers` (que retorna cópias). O treinamento Python preserva os pesos originais caso uma execução falhe.

## Evidências

- **P0–P3:** CI, testes matemáticos/backprop, aprendizado XOR, persistência e paridade Python × JavaScript.
- **P4–P5:** benchmarks Node, smoke test real Chrome headless e extensão Manifest V3 isolada.
- **P6:** integridade do treinamento com commit seguro e validação numérica.
- **P7:** conjuntos artificiais de treino/teste separados, ruído e comparação com métodos tradicionais.
- **P8:** contrato API, documentação, licenciamento, validação de ambiente e revisão final.

Resultados de generalização estão restritos às distribuições artificiais de [evaluation.md](evaluation.md); latência em runner é apenas referência e está documentada em [performance.md](performance.md).

## Segurança, portabilidade e não objetivos

O runtime não usa rede, telemetria, Node, DOM, Cloudflare, eval ou APIs pagas. A origem/armazenamento do modelo, extração de atributos e uso da previsão são do consumidor.

Não comprovados: Cloudflare Workers, Safari, Firefox, mobile, lifecycle extensivo de service worker ou aplicação real. Não incluídos: LLM, transformers, WebGPU, autoaprendizado, integração com projetos específicos e publicação npm. Extensões futuras requerem testes e evidência concreta.
