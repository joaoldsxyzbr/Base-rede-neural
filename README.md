# Neural Core

Motor neural genérico, portátil e independente, construído em **JavaScript puro**, sem dependências obrigatórias, APIs pagas ou telemetria.

## Objetivo

Fornecer uma base para construir, treinar, avaliar, salvar e executar redes neurais em aplicações diferentes. **As aplicações se adaptam ao Neural Core, nunca o contrário.**

## Diretrizes

1. Simplicidade, robustez e qualidade.
2. Universalidade e independência.
3. Arquitetura modular e extensível.
4. Portabilidade e compatibilidade.
5. Desempenho e custo zero.
6. Aprendizado real e qualidade verificável.
7. Segurança, privacidade e confiabilidade.

## Primeira versão (proposta, ainda não implementada)

- Rede MLP com camadas densas, inicialmente para classificação e regressão simples.
- Inferência e treinamento com `Float32Array` e APIs nativas de JavaScript.
- Treinamento supervisionado com retropropagação e SGD.
- Pesos serializáveis em formato versionado e validado.
- Testes reproduzíveis de cálculos, treinamento, persistência e portabilidade.
- Nenhuma especialidade de produto dentro do núcleo.

**Estado:** arquitetura definida; implementação e testes ainda pendentes.

Confira [a arquitetura e as etapas propostas](docs/architecture.md).
