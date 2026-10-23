# P5 — Validação isolada no Manifest V3

## Objetivo e limites

Provar que o motor JS atual pode ser **incluído localmente** em uma extensão Chrome Manifest V3 e funcionar no popup e no service worker. O Neural Core **não** ganha dependência de Chrome e nenhum outro projeto ou extensão é alterado. Os arquivos nesta pasta são um harness de teste, não um novo produto.

## Como testar

```bash
node --test tests/extension-mv3.test.js
node scripts/extension-smoke.mjs
```

O primeiro comando roda testes estáticos sobre o manifesto e as origens dos módulos. O segundo requer Chromium ou Chrome for Testing com suporte a extensões `--load-extension` (ou `CHROME_BIN` com caminho apropriado).

O runner cria um diretório temporário, copia os arquivos **originais** `src/index.js`, `src/model.js`, `src/network.js` e `tests/fixtures/parity.json`, e carrega uma extensão MV3 desempacotada. O processo de verificação usa o protocolo nativo CDP com relógio real, sem acelerar artificialmente os timers do service worker. Seu popup compara 19 previsões Python × JS e faz ida e volta `serializeModel`/`loadModel`. Em seguida envia uma mensagem ao **service worker de módulo**, que roda `predict` com entrada `[4,5]` e confirma saída `24`. O runner confere um resultado `PASS` real do DOM carregado a partir de `chrome-extension://`. O resultado é inválido se a extensão não carregar ou se o service worker não responder.

## Segurança e dependências

- Nenhuma permissão de site, `storage`, `tabs`, `activeTab` ou `webRequest`.
- Nenhuma telemetria, armazenamento de dados ou código hospedado remotamente.
- Os modelos de teste acompanham a própria extensão; não há `fetch` para servidores externos.
- A chave no `manifest.json` é apenas **chave pública de desenvolvimento**, para fornecer um ID de teste previsível. Não contém chave privada nem credencial.
- O Chrome é baixado somente no CI, para o job de teste. O downloader usa biblioteca padrão Python, HTTPS e restringe a origem ao serviço oficial de downloads do Google. Não é uma dependência de inferência.

O Chrome oficial com marca passou a limitar flags de carregamento de extensões a partir do Chrome 139; por isso o CI usa **Chrome for Testing**, em vez de presumir que `google-chrome` do runner aceitará `--load-extension`.

## O que esse teste não prova

Não testa o comportamento visual da extensão de modo escuro, scripts de conteúdo, permissões de host, longos períodos de suspensão/reinício do service worker, navegadores diferentes, desempenho em celulares, acesso à Chrome Web Store ou compatibilidade com Cloudflare Workers. Essas verificações continuam independentes.

## Evidência

A aprovação em `node --test` confirma apenas a **estrutura**; a validação de MV3 requer que o job `Extensão Manifest V3 / Chrome for Testing` do GitHub Actions efetivamente passe. Não marcar este passo como completo sem inspecionar esse job.

## Resultado verificado — 23/10/2026

**P5 aprovado no GitHub Actions.** [Execução completa](https://github.com/joaoldsxyzbr/Base-rede-neural/actions/runs/40007501913) do commit `a65719fda3bf89a3c4ff2d4cd09c31772f2bbee4`: **8 de 8 jobs concluídos com sucesso**.

O job `Extensão Manifest V3 / Chrome for Testing` instalou Chrome for Testing oficialmente, carregou a extensão desempacotada e registrou a linha real:

```text
PASS: MV3 popup (19 cases) and service worker
```

Esse resultado comprova a execução de inferência no popup e no service worker MV3 **na configuração do Chrome for Testing do CI**, com os modelos de referência. Não comprova a extensão Dark Reader, Worker Cloudflare, execução em dispositivos móveis, ciclo de vida prolongado ou a publicação na Web Store.

## Correção de intermitência (23/10/2026)

Após a primeira execução aprovada, uma repetição do CI falhou porque o script antigo usava `--dump-dom` com relógio virtual: o temporizador do popup podia vencer antes de o worker iniciar. O novo teste usa o **Chrome DevTools Protocol**, embutido no Chrome e acessado pelo `WebSocket` nativo do Node, com espera em tempo real, tempo limite explícito e verificação do resultado da extensão. A falha anterior foi registrada em [run 40007815157](https://github.com/joaoldsxyzbr/Base-rede-neural/actions/runs/40007815157). Essa melhoria não altera o motor neural nem oculta falhas persistentes do service worker.
