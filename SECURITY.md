# Segurança e privacidade

O Neural Core é uma base local. O código de inferência em `src/` não envia dados, não tem telemetria, não executa código incluído nos modelos e não precisa de serviços de terceiros. O Chrome, Playwright e Wrangler usados no CI são **ferramentas de teste**, não dependências de consumidores. Playwright e Wrangler são instalados por `npm ci` a partir de um lockfile de integridade em `ci-tools/`, reduzindo alterações não rastreadas na árvore de dependências de teste.

Use arquivos JSON v1 de fontes confiáveis e verifique sua origem/integridade. É responsabilidade do aplicativo controlar modelos, atributos, permissões e armazenamento. Em JavaScript, modelos passados como objetos com getters ou Proxy podem executar comportamentos do **próprio objeto**; prefira interpretar JSON ao receber dados de fontes não confiáveis.

Para comunicar vulnerabilidades, use o recurso **Private Vulnerability Reporting** do GitHub, se habilitado, ou um canal privado definido com o mantenedor. Não publique dados sensíveis ou instruções de exploração em issues públicas.

Testes em Node, Chrome, extensão MV3, Firefox/WebKit Linux e `workerd` local não substituem auditoria formal de segurança. Safari nativo, dispositivos móveis físicos e Workers publicados continuam não comprovados. No JSON v1 o campo reservado opcional `operator` precisa ser `dense`; valores desconhecidos são rejeitados no JS e no Python.
