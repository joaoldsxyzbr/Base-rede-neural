# Segurança e privacidade

O Neural Core é uma base local. O código de inferência em `src/` não envia dados, não tem telemetria, não executa código incluído nos modelos e não precisa de serviços de terceiros. O Chrome baixado pelo CI é uma **ferramenta temporária de testes**, não uma dependência de consumidores.

Use arquivos JSON v1 de fontes confiáveis e verifique sua origem/integridade. É responsabilidade do aplicativo controlar modelos, atributos, permissões e armazenamento. Em JavaScript, modelos passados como objetos com getters ou Proxy podem executar comportamentos do **próprio objeto**; prefira interpretar JSON ao receber dados de fontes não confiáveis.

Para comunicar vulnerabilidades, use o recurso **Private Vulnerability Reporting** do GitHub, se habilitado, ou um canal privado definido com o mantenedor. Não publique dados sensíveis ou instruções de exploração em issues públicas.

Testes em Node, Chrome e extensão MV3 não equivalem a uma auditoria formal de segurança e não garantem funcionamento em Cloudflare Workers, Firefox, Safari ou dispositivos móveis.
