# Roadmap — Neural Core

Esta sequência prepara a **primeira versão reutilizável**, não encerra o desenvolvimento.

| Etapa | Objetivo | Situação |
| --- | --- | --- |
| P0 | CI JS | Concluído |
| P1 | Testes matemáticos, validação e isolamento JS | Concluído |
| P2 | Treinamento MLP Python e retropropagação | Concluído |
| P3 | Paridade Python × JavaScript | Concluído |
| P4 | Benchmark e navegador Chrome real | Concluído |
| P5 | Extensão independente Manifest V3 | Concluído |
| P6 | Treinamento seguro e números extremos | Concluído |
| P7 | Generalização sintética e comparação tradicional | Concluído |
| P8 | API, documentação, licença, ambiente CI e revisão | Concluído — CI de fechamento aprovado |
| P9 | Integridade de treinamento, persistência e contratos de compatibilidade | Concluído — testes de regressão e documentação na PR #1 |
| P10 | Mini-batches, SGD com momentum, AdamW, ReLU, softmax e classificação | Concluído na PR #2 após CI aprovado |
| P11 | Validação independente, early stopping, métricas e experimentos controlados | Concluído — PR #3 com CI aprovado |
| P12 | Inferência em lote e buffers temporários, paridade e benchmarks comparativos | Concluído — PR #4, CI e medições documentadas |

**Critério final:** pacote v0.1.0, API documentada, licença MIT, SHA-256 do Chrome for Testing fixado, suíte verde e auditoria de limites registrada em `docs/audit.md`. Não há publicação npm nem etiqueta Git criada automaticamente.

Após a v0.1.0: avaliar integrações em projetos consumidores, Cloudflare Workers, celulares, mais arquiteturas e datasets reais **mediante necessidade e métricas reproduzíveis**. Nenhuma plataforma ou aplicação deve ser obrigatória para a base.
