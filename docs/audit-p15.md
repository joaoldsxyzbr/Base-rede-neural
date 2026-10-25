# Auditoria P15 — Consolidação

## Resultado verificado

- Versão técnica da PR #7: [CI 40199874951](https://github.com/joaoldsxyzbr/Base-rede-neural/actions/runs/40199874951) — **11/11 jobs aprovados**.
- O núcleo permanece JavaScript ES Modules, sem dependências de execução. O treinamento usa a biblioteca padrão Python.
- O campo reservado `operator`, quando explícito, precisa ser `dense` em JS e Python. A ausência do campo mantém o mesmo significado e `serializeModel` continua emitindo JSON v1 sem esse campo.
- Testes determinísticos de topologias, entrada inválida, paridade entre lote e previsão individual e rollback de treinamento.
- Benchmark de stress: 12 lotes de 4.096 amostras (49.152 inferências). Registra heap e RSS apenas para diagnóstico, sem limite de aprovação instável.
- Ferramentas oficiais de teste Playwright e Wrangler têm versões transitivas registradas em `ci-tools/package-lock.json`; instalação pelo CI usa `npm ci`. O pacote principal não ganhou dependências.

## Ação administrativa pendente: proteção da main

A leitura da API do GitHub antes do P15 mostrou `protected: false` e `rulesets: []`. A integração GitHub deste chat **não oferece ação de edição de regras de proteção**. Assim, a exigência administrativa não foi aplicada nem considerada concluída. Configurar no GitHub conforme [branch-protection.md](branch-protection.md) e confirmar a proteção efetiva.

## Limitações que continuam válidas

- Testes gerativos determinísticos não substituem fuzzing independente, auditoria de segurança ou verificação formal.
- Memória medida em runner compartilhado não demonstra ausência de vazamento em sessões prolongadas.
- Cloudflare `workerd` foi testado localmente, sem publicação real; WebKit Linux e emulação não substituem Safari/iPhone reais.
- Métricas de generalização em dados sintéticos não comprovam desempenho em dados reais. Consumidores precisam de validação por pessoa/grupo/tempo, baselines e teste intocado.
- A versão `0.1.0` não implica tag Git ou publicação npm. O registro histórico P8 permanece em [audit.md](audit.md).
