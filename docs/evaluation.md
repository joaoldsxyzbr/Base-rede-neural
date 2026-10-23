# Avaliação reproduzível — P7

## Objetivo e limites

Evidenciar **aprendizado em dados não vistos** e comparar a MLP com métodos tradicionais simples. Ambos os estudos usam dados inteiramente artificiais, sementes determinadas e **sem dependências externas**. São testes diagnósticos do algoritmo, não avaliação de uma aplicação real nem demonstração de superioridade universal de redes neurais.

Executar na raiz do repositório:

```bash
python -m unittest discover -s training_tests -v
python -m evaluation.benchmark
```

A execução imprime JSON com resultados por semente, resumos estatísticos e hiperparâmetros fixos. O CI usa Python 3.11/3.13 nos testes e publica a avaliação em um job próprio com Python 3.13.

## Experimento 1 — Classificação não linear

Problema artificial: duas entradas numéricas com rótulo 1 quando possuem sinais diferentes (quatro quadrantes, distribuição balanceada). Coordenadas latentes têm magnitude de 0,25 a 1 para evitar ambiguidade excessiva junto às fronteiras.

- Treino: 96 amostras, ruído gaussiano nas entradas com desvio-padrão 0,05.
- Teste: **128 amostras novas**, geradas com outra semente, sem sobreposição com o treino.
- Teste ruidoso: as **mesmas 128 amostras latentes** do teste, mantendo seus rótulos, com ruído gaussiano de desvio-padrão 0,35 nas entradas observadas.
- MLP: [2, 8, 1], tanh + sigmoid, 350 épocas, SGD 0,3.
- Referenciais: rede com **classificador linear sigmoide** [2, 1], SGD com mesma configuração; probabilidade constante estimada pela proporção majoritária observada no treino.
- Métricas: acurácia com corte em 0,5 e Brier (média do quadrado entre probabilidade estimada e alvo).
- Repetições: sementes 7, 19 e 31; reportados médias e desvios-padrão populacionais sobre as três repetições.

## Experimento 2 — Regressão linear

Dados artificiais: `y = 2x + 1 + ε`, com `x` uniforme entre -1 e 1 e ruído gaussiano independente (desvio-padrão 0,08).

- Treino: 96 observações. Teste: 128 observações **independentes**.
- Modelo Neural Core: camada [1, 1] com saída linear, SGD 250 épocas e taxa 0,1.
- Referenciais: solução analítica de **mínimos quadrados ordinários** e predição constante da média dos alvos de treino.
- Métricas: MSE e MAE em treino e teste.
- Mesmas três sementes e mesmas amostras de treino/teste para todos os métodos em cada repetição.
- Espera-se que mínimos quadrados sejam tão bons quanto a rede linear, e não que a rede seja superior.

## Resultados iniciais — Python 3.13, execução local em 23/10/2026

| Classificação (média de três sementes) | Acurácia teste limpo | Acurácia teste com ruído |
| --- | ---: | ---: |
| MLP 2→8→1 | **100,00%** | **89,58%** |
| Classificador linear | 48,96% | 48,44% |
| Maioria | 50,00% | 50,00% |

| Regressão (média de três sementes) | MSE de teste |
| --- | ---: |
| Rede linear 1→1 | 0,006569 |
| Mínimos quadrados | **0,006558** |
| Média constante | 1,393230 |

As taxas de 100% do primeiro experimento refletem a **distribuição sintética separável e com margem**; não são taxas de acerto esperadas em dados reais. No caso linear, o método tradicional apresentou erro ligeiramente menor, como esperado.

## Limitações e política de qualidade

- As amostras de teste não participam das atualizações de peso nem da escolha de checkpoints. Hiperparâmetros são fixos para o protocolo; os experimentos não usam otimização automática de hiperparâmetros.
- Três sementes são uma verificação inicial de estabilidade, **não** um intervalo de confiança estatístico nem garantia de generalização fora da distribuição estudada.
- Comparações são entre métodos pequenos implementados de maneira transparente, não com bibliotecas de produção otimizadas.
- Mudanças futuras no algoritmo devem ser avaliadas com o mesmo protocolo e também com novos conjuntos de avaliação independentes para evitar ajustar o projeto ao benchmark.
- Os limites numéricos de teste são um controle de regressão apenas para os cenários determinísticos; não são metas gerais para usuários ou projetos consumidores.
- P7 não modifica `src/`, não cria treinamento automático nos aplicativos e não depende de APIs ou datasets externos.
