# Portabilidade verificada — P13

## Princípio

O Neural Core é executado pelos **mesmos arquivos** `src/index.js`, `src/model.js` e `src/network.js` em todos os ambientes. Nenhum projeto consumidor, backend, serviço Cloudflare, navegador ou SDK é obrigatório para **inferência**. Python continua opcional e destinado ao treinamento offline.

O formato público segue `formatVersion: 1`. O runtime não importa Node, Cloudflare, DOM, Playwright, Wrangler ou bibliotecas externas.

## Matriz de evidências e seus limites

| Ambiente | Verificação no CI | Limite da afirmação |
| --- | --- | --- |
| Node.js 22 e 24 | Testes nativos `npm test` | Não comprova outras versões ou dispositivos |
| Chrome headless | Página ESM real com previsões, paridade, ReLU/softmax e `predictBatch` | Não significa cobertura de todos os navegadores Chromium |
| Chrome Manifest V3 | Popup e service worker com o Core real | Não comprova ciclo longo/suspensão-retomada do worker |
| Firefox (binário Playwright Linux) | A mesma página ESM, com engine Firefox real | Binário do Playwright pode diferir da versão Firefox do usuário |
| WebKit (binário Playwright Linux) | A mesma página ESM com engine WebKit | **Não é Safari** no macOS ou no iPhone |
| WebKit com perfil de iPhone emulado | Mesmo teste com dimensões e contexto emulados | **Não é dispositivo iOS real**, não comprova performance/touch em hardware |
| Cloudflare `workerd` local | Worker de teste com imports do Core, inferência, batch, serialização e validação | **Não houve deploy** nem teste de cotas de CPU/memória no serviço remoto |
| Safari nativo, Firefox estável distribuído ao usuário, Android e iOS físicos | **Ainda não testados** | Requer ambientes reais específicos |

O CI P13 adiciona dois jobs: `Portabilidade / Firefox e WebKit` e `Portabilidade / Cloudflare workerd local`. Uma execução do CI verde é requisito antes de tratar cada linha nova como evidência efetiva; a documentação registra *o que* os jobs testam, não um selo universal de compatibilidade.

## Reproduzir localmente

O runtime e o treinamento mantêm zero dependências externas. No P15, os pacotes oficiais de teste têm lockfile em `ci-tools/package-lock.json` e são instalados com `npm ci`; `node_modules` é apenas um link simbólico temporário no ambiente de teste. Só os **testes de portabilidade** instalam ferramentas oficiais de seus fornecedores, temporariamente no ambiente de CI ou de desenvolvimento.

```bash
# Em um ambiente Linux com Node.js 24
PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm ci --prefix ci-tools --no-audit --no-fund
ln -s ci-tools/node_modules node_modules
ci-tools/node_modules/.bin/playwright install --with-deps firefox webkit
node scripts/cross-browser-smoke.mjs
```

Para testar `workerd` local, em ambiente de teste separado:

```bash
# Se o setup acima já foi executado, reutilize-o sem reinstalar:
PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm ci --prefix ci-tools --no-audit --no-fund
ln -s ci-tools/node_modules node_modules
WRANGLER_SEND_METRICS=false CLOUDFLARE_SEND_METRICS=false node scripts/worker-smoke.mjs
```

O script do Worker inicia somente `wrangler dev --local` em `127.0.0.1` com porta aleatória e `compatibility_date=2026-10-24`, chama seu endpoint e encerra o processo. Nenhuma publicação, login, binding remoto ou acesso a dados de usuário é necessário. `WRANGLER_SEND_METRICS=false` desliga as métricas opcionais do Wrangler; os executáveis de teste podem precisar de download oficial na instalação.

Os testes de Firefox/WebKit servem apenas arquivos explicitamente permitidos em `127.0.0.1` e bloqueiam solicitações externas do navegador durante a verificação. Eles não são dependências da biblioteca.

## Garantias e próximos experimentos

- Rejeição de formatos desconhecidos e manutenção de `float32` são verificadas em mais engines, preservando os limites definidos em [API](api.md).
- A plataforma que hospeda um aplicativo é responsável por lidar com ciclo de vida, disponibilidade de CPU/memória, origem e segurança de modelos.
- Teste em Safari/macOS e dispositivos físicos requer runners e aparelhos reais; evitar confundir emulação de viewport e WebKit Linux com compatibilidade atestada.
- Testes no `workerd` local não demonstram custos, limites do plano Cloudflare nem ausência de incompatibilidades com configurações de produção.
- Nenhuma arquitetura ou operador novo foi adicionado no P13. Futuras melhorias continuam condicionadas a dados e critérios reproduzíveis.

## Evidência inicial — 24/10/2026

A PR #5 foi executada com **11/11 jobs aprovados** em [GitHub Actions 40122047553](https://github.com/joaoldsxyzbr/Base-rede-neural/actions/runs/40122047553), incluindo Node 22/24, Python 3.11/3.13, paridade, Chrome, Manifest V3, avaliação científica, benchmark e os dois novos jobs de portabilidade.

O job do Firefox/WebKit registrou `PASS: 22 previsões no navegador` para Firefox, WebKit Linux e WebKit com emulação móvel. O job de `workerd` registrou `PASS` para inferência, lote, serialização e validação local. As verificações cobrem **somente** as versões/ambientes executados nessa data; não constituem teste em dispositivos físicos nem validação de Workers em produção.
