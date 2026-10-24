# Compatibilidade e evolução do formato de modelos

O Neural Core mantém versões distintas para o **pacote** (`package.json`, atualmente `0.1.0`) e para o **arquivo de modelo** (`formatVersion: 1`). Evoluir um deles não implica alterar o outro. A v0.1.0 não é publicada no npm.

## Compromissos do formato v1

- Modelos v1 já válidos continuam carregáveis na API JavaScript existente: `loadModel`, `predict`, `serializeModel`.
- O contrato suporta camadas densas, ativações legadas `linear`, `tanh`, `sigmoid` e, desde P10, `relu` e `softmax` (apenas na saída com pelo menos duas classes), pesos na ordem [saída][entrada] e vieses de saída. Nenhum modelo v1 antigo mudou de significado.
- A API não depende de plataforma, projeto consumidor, rede, telemetria nem dependências adicionais.
- Campos extras em modelos v1 são ignorados pelo leitor JavaScript e não reaparecem após `serializeModel`. Eles **não** podem ser usados para alterar o significado dos campos conhecidos.
- Versões de arquivo desconhecidas, dimensões inválidas, parâmetros não finitos e números não representáveis em float32 são rejeitados; não há conversão silenciosa de v2 para v1.
- O treinamento Python usa float64 internamente, mas antes de confirmar novos pesos valida se o modelo é exportável em float32. Treinamentos inválidos preservam os pesos anteriores.
- `save_model` usa um temporário no mesmo diretório e `os.replace`, evitando truncamento direto de um modelo preexistente. A operação pode falhar por permissões, bloqueios ou problemas de I/O; não oferece transações entre arquivos nem garantia de sobrevivência a toda queda de energia.

## Introdução de um futuro formato v2

1. Propor um novo contrato explícito para qualquer arquitetura, operador ou alteração de significado incompatível com v1. Não reutilizar `formatVersion: 1` para semânticas novas.
2. Acrescentar suporte de leitura de v2 **sem retirar** o leitor v1; o exportador permanece em v1 por padrão enquanto os consumidores dependerem dele.
3. Criar fixtures de paridade Python×JS, testes de serialização, limites, modelos malformados e migração quando necessária.
4. Demonstrar uma aplicação ou experimento independente que justifique o formato e medir regressões antes de mudar padrões.
5. Documentar o período de suporte e qualquer quebra de API do pacote antes de lançar uma nova versão. Não existe compromisso de suporte perpétuo sem testes.

## Matriz de compatibilidade verificada

O CI cobre Node.js 22/24, Python 3.11/3.13, Chrome headless e extensão Chrome Manifest V3 (popup/service worker). Isso não comprova Firefox, Safari, Cloudflare Workers, iOS, Android ou runtimes sem navegador. Ampliar a matriz exige testes reais nesses ambientes.

A API e os limites atuais estão em [api.md](api.md); a visão de arquitetura, em [architecture.md](architecture.md).
