# Avaliação reproduzível — P7 e P11

## Objetivo e limites

Evidenciar **aprendizado em dados não vistos** e comparar a MLP com métodos tradicionais simples. Ambos os estudos usam dados inteiramente artificiais, sementes determinadas e **sem dependências externas**. São testes diagnósticos do algoritmo, não avaliação de uma aplicação real nem demonstração de superioridade universal de redes neurais.

Executar na raiz do repositório:

```bash
python -m unittest discover -s training_tests -v
python -m evaluation.benchmark
python -m evaluation.quality
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

## P11 — Protocolo de validação e teste

`python -m evaluation.quality` emite um segundo relatório JSON (`protocolVersion: 2`) **separado** dos experimentos P7, preservando o protocolo anterior para comparações futuras.

- Problema: classificação sintética dos quatro quadrantes, equilibrada por classe.
- Divisão padrão: 96 observações de treinamento (ruído gaussiano 0,05), 64 de validação (0,05) e 128 de teste (0,15). Sementes de geração independentes para cada divisão; o relatório registra todas as quantidades e parâmetros.
- Redes comparadas sob as **mesmas amostras** por semente: MLP [2,8,1] com SGD, momentum ou AdamW; classificador linear sigmoide [2,1]; e probabilidade constante estimada no conjunto de treinamento.
- Hiperparâmetros fixados antes da avaliação: 160 épocas no máximo, 20 de paciência, mini-batch 8, entropia cruzada binária; taxas por otimizador documentadas no JSON.
- Seleção do checkpoint: exclusivamente perda de validação (a época 0 é candidata). Nenhum modelo vê exemplos de teste durante atualizações, seleção de época ou escolha de hiperparâmetros.
- Métricas: acurácia, precisão, recall, F1, Brier e matriz de confusão por divisão e por semente; resumo com média e desvio-padrão populacional. Se não houver positivos previstos, precisão/recall/F1 usam 0 em vez de divisão por zero.
- Sementes padrão 7, 19 e 31. **Três repetições não representam intervalo de confiança** e não justificam afirmar que um otimizador é universalmente melhor.
- A separação de entradas idênticas e os diferentes geradores reduzem vazamento nos dados sintéticos, mas não substituem uma divisão por grupos/tempo em bases reais.
- Novos relatórios são comparações diagnósticas; o processo não ajusta parâmetros depois de observar resultados de teste.

A avaliação P11 não altera `src/`, o formato JSON de modelos ou os experimentos P7. Não usa datasets privados, bibliotecas de terceiros nem recursos pagos.

## Resultado inicial do P11 (24/10/2026)

Execução [GitHub Actions 40092079823](https://github.com/joaoldsxyzbr/Base-rede-neural/actions/runs/40092079823), no job de qualidade científica com Python 3.13. São médias de três sementes (7, 19 e 31), obtidas no **teste sintético ruidoso**, com hiperparâmetros fixados antes de observar o teste.

| Método | Acurácia média no teste | F1 médio no teste | Brier médio no teste |
| --- | ---: | ---: | ---: |
| MLP + SGD | 98,70% | 0,9870 | 0,00809 |
| MLP + momentum | 99,22% | 0,9921 | 0,00720 |
| MLP + AdamW | 99,48% | 0,9948 | 0,00455 |
| Classificador linear | 50,26% | 0,5081 | 0,25023 |
| Probabilidade constante | 50,00% | 0,6667 | 0,25000 |

A probabilidade constante de 0,5 prevê a classe positiva no limiar de decisão, daí seu F1 aparentemente alto apesar de acurácia de apenas 50%. O conjunto foi desenhado para conter uma fronteira não linear; a tabela **não prova** que AdamW supere os demais fora desse experimento, nem constitui benchmark de aplicações reais. Alterar hiperparâmetros em resposta a esse resultado requer reservar um novo teste final independente.


## P16 — Primeiro conjunto de dados reais: Iris (classificação multiclasse)

Executar `python -m evaluation.real_data`. A avaliação usa apenas a biblioteca padrão e uma cópia **versionada** de `evaluation/data/iris.csv`. Não há download durante CI nem treinamento. O dataset contém 150 observações de flores reais, 4 medidas numéricas e 3 espécies. Cada conjunto de dados tem limitações de amostragem: este caso é **pequeno e clássico**, não simula operações reais em sites/extensões, e não garante generalização em produção.

**Origem e atribuição da base de dados:** Fisher, R. (1936). *Iris* [Dataset]. UCI Machine Learning Repository, DOI [10.24432/C56C76](https://doi.org/10.24432/C56C76). [UCI Iris nº 53](https://archive.ics.uci.edu/dataset/53/iris), licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). A cópia CSV sem cabeçalho corresponde às amostras distribuídas também em [Plotly/datasets — iris.csv](https://github.com/plotly/datasets/blob/master/iris.csv), consultadas em 25/10/2026. Os 150 registros originais foram preservados. Esta seção fornece o crédito e a indicação da licença do dataset, independentemente da licença MIT do código.

**Protocolo fixo antes de analisar o teste:**

- Repetições com sementes 7, 19 e 31. Em cada repetição, divisão estratificada próxima de 60% treino, 20% validação e 20% teste; vetores de atributos *idênticos* ficam no mesmo conjunto. O CSV contém exemplos repetidos. Relações semânticas não detectáveis não são controladas, e os três testes compartilham observações entre repetições diferentes; não se trata de três datasets independentes.
- Médias e desvios-padrão de normalização são calculados **somente** com o treino e depois aplicados à validação/teste. Não ajustar preparação, arquitetura ou hiperparâmetros em função dos resultados finais.
- MLP fixa de 4→8→3, tanh + softmax, perda CCE, AdamW (taxa 0,04, decaimento 0), mini-batch 12, 160 épocas no máximo, paciência de validação 20 e checkpoint de menor perda de validação.
- Referências não neurais sob as mesmas partições: **centróide mais próximo** (médias por classe calculadas somente do treino) e **classe majoritária** do treino. Ambos usam as mesmas medidas normalizadas quando aplicável.
- Saída JSON imprime por semente quantidades de exemplos, acurácia, F1 macro e matriz de confusão no teste, época selecionada e resumos descritivos. Métricas são **diagnósticas**, não um ranking universal nem limite de aprovação do CI.

Os testes `training_tests/test_real_data.py` exigem integridade do dataset, reprodutibilidade das divisões, agrupamento de entradas duplicadas, isolamento da normalização, métricas e execução integrada. O CI testa a avaliação, mas não declara a MLP superior ao baseline. Para aplicações futuras, usar dados pertinentes ao domínio, divisão por usuário/grupo/tempo quando apropriado, mais amostras e teste final intocado.
