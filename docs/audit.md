# Auditoria de consolidação — Neural Core v0.1.0 (histórico P8)

> Documento histórico. Para o estado recente, consultar [auditoria P15](audit-p15.md) e [matriz de portabilidade](portability.md).

**Escopo:** revisão da versão inicial em `main` após P0–P8. A evidência é código, testes reproduzíveis e logs do GitHub Actions — não auditoria externa de segurança nem validação de aplicativos consumidores.

## Inventário e resultados verificados

- API pública JS: `loadModel`, `predict`, `serializeModel`; contrato de modelo JSON `formatVersion:1`, testes de importação por nome próprio e de isolamento dos parâmetros.
- Algoritmos: MLP feed-forward com ativações `linear`, `tanh`, `sigmoid`. Python offline usa retropropagação/SGD reproduzíveis; validação dos gradientes por diferenças finitas; preservação integral do modelo original quando o treinamento diverge.
- Dados: sem telemetria, servidores ou provedores obrigatórios no runtime; parâmetros internamente protegidos; modelos inválidos/dimensões abusivas rejeitados.
- Paridade: fixtures Python e job que gera referência fresca e a compara com execução JS. Testes reais no Chrome e extensão MV3 com popup e service worker.
- Qualidade científica: avaliação com dados sintéticos de treino/teste separados, ruído e métodos tradicionais; nenhuma alegação de precisão em tarefas reais.
- CI intermediário do P8: [run 40016217809](https://github.com/joaoldsxyzbr/Base-rede-neural/actions/runs/40016217809), **9/9 jobs aprovados**, **26/26 testes JS**, **27/27 Python**, paridade, benchmarks, Chrome e Manifest V3. O Chrome usado teve SHA-256 `e6c94490f0b6b337225aeb2d2a5034858516b7017356980ba94f5176d8774a50`.
- A revalidação final foi concluída no [GitHub Actions 40016584472](https://github.com/joaoldsxyzbr/Base-rede-neural/actions/runs/40016584472), commit `e0459610584d0edba502d18c3697cf92d8ce7b68`: **9/9 jobs aprovados**, **26/26 testes JavaScript**, **28/28 testes Python**. O instalador verificou o SHA-256 estritamente e o smoke MV3 registrou `PASS: MV3 popup (19 cases) and service worker`.

## Organização e contrato

`README.md` foi atualizado para o estado atual, `docs/architecture.md` perdeu descrições de etapas antigas, `docs/api.md` documenta assinaturas/erros/limites, `ROADMAP.md` registra P0–P8. `SECURITY.md` esclarece política de privacidade e reporte; `LICENSE` adota MIT (titular identificado pelo usuário do repositório). O `package.json` continua `private: true` e fornece somente a entrada ESM; não há publicação npm.

## Limitações remanescentes (não bloqueiam o escopo v0.1.0)

1. Não foram executados testes reais em Cloudflare Workers, Firefox, Safari, celulares ou hardware restrito; a portabilidade comprovada limita-se aos ambientes do CI.
2. Testes de aprendizado e generalização são sintéticos; aplicações reais exigem dados independentes, métricas e comparação adequadas ao domínio.
3. Resultados de latência/memória provêm de runners compartilhados; não garantem velocidade/consumo em extensões ou celulares.
4. O Chrome for Testing foi fixado por versão/digest para o job MV3. Node e Python no GitHub Actions estão fixados por **linhas principais**, não versões de patch; o runner Ubuntu também é atualizado ao longo do tempo. Portanto, o ambiente não é bit a bit imutável.
5. A versão `0.1.0` existe no `package.json`; criar uma tag Git `v0.1.0` ou release é uma operação separada. **Não afirmar que existe sem verificar.**
6. Sem cobertura formal de fuzzing, longas sessões de MV3 nem auditoria independente de supply chain. APIs de dados JavaScript podem executar getters de objetos recebidos; preferir modelos JSON de origem validada.

## Decisão

A arquitetura pode ser reutilizada em projetos experimentais controlados, mantendo os limites claros e o contrato versionado. Novas arquiteturas e otimizações são evolução futura, fora do P8. A aprovação final foi confirmada após a mudança de SHA-256. Nenhum outro escopo/plataforma é assumido como validado.
