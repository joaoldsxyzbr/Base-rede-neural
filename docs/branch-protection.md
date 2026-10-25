# Proteção administrativa da branch main

**Estado verificado antes do P15:** `protected: false`, nenhum ruleset. O plugin GitHub usado aqui não possui ação para editar proteção de branches. É necessário configurar manualmente no repositório (ou usar uma ferramenta administrativa autorizada).

1. Abrir **GitHub → Base-rede-neural → Settings → Rules → Rulesets → New branch ruleset** (ou **Settings → Branches → Add branch protection rule**).
2. Criar `main-ci-protection` com alvo `main`, enforcement **Active**.
3. Habilitar **Require a pull request before merging**. Com um único mantenedor, não obrigar revisão de um segundo usuário sem necessidade: isso poderia bloquear o projeto.
4. Habilitar **Require status checks to pass** e selecionar os checks que aparecem na PR: JavaScript (Node 22/24), Treinamento (Python 3.11/3.13), Integração Python×JS, Chrome, Benchmark, MV3, Avaliação científica, Firefox/WebKit e `workerd` local.
5. Bloquear exclusão e force push na `main`. Evitar exceções amplas de bypass.
6. Salvar e **verificar** a regra ativa. Uma PR com falha de CI deve ficar impedida de merge e um push direto deve ser bloqueado.

Notas: CI verde **sem regra de proteção** não impede merge indevido. Exigir checks que não existam ou renomeados pode bloquear PRs. Não colocar tokens pessoais no repositório para configurar regras.
