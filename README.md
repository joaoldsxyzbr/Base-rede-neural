# Neural Core

Base neural **genérica, independente e reutilizável**, sem acoplamento a aplicações ou plataformas. Os projetos se adaptam ao Neural Core, e não o contrário.

## Linguagens e responsabilidades

- **JavaScript puro (ES Modules):** núcleo de **inferência** portátil para navegadores, extensões, Node.js e outros ambientes compatíveis; sem dependências externas obrigatórias.
- **Python:** ferramentas separadas de **treinamento, experimentação, avaliação e exportação** dos modelos. Não é exigido em tempo de execução nos aplicativos.
- **Formato compartilhado:** arquitetura, ativações e pesos exportados em formato versionado e validado; previsões JavaScript devem ser comparadas com referências produzidas pelo treino Python.

A primeira versão (v0.1) terá uma MLP pequena para classificação e regressão, **não** um modelo de conversação generativa. Treinamento e inferência serão separados; não haverá treinamento automático em produção.

## 7 regras fundamentais

1. **Simplicidade, robustez e qualidade:** código claro, seguro e fácil de manter.
2. **Universalidade e independência:** nenhuma lógica específica de projeto no núcleo.
3. **Arquitetura modular e extensível:** componentes com responsabilidades bem definidas.
4. **Portabilidade e compatibilidade:** JavaScript para execução; Python para ferramentas de treinamento.
5. **Desempenho e custo zero:** modelos compactos, processamento eficiente, sem servidor ou API paga obrigatórios.
6. **Aprendizado real e qualidade verificável:** testes reproduzíveis, métricas e comparação com alternativas tradicionais.
7. **Segurança, privacidade e confiabilidade:** processamento local, validação rigorosa, sem telemetria e sem comprometer compatibilidade.

## Estado

**Arquitetura planejada; nenhuma implementação ou treinamento concluído.**

Consulte [a arquitetura, o formato dos modelos e as etapas de desenvolvimento](docs/architecture.md).
