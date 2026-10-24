# Treinamento avançado — P10

Treinamento offline em Python (stdlib) e inferência local em JavaScript puro, sem dependências novas. Todos os métodos são supervisionados e usam redes densas.

## Ativações e inicialização

- `linear`, `tanh`, `sigmoid`: funcionamento e inicialização Xavier legados preservados.
- `relu`: `max(0,z)`, derivada 0 em z=0; inicialização He uniforme no intervalo ±sqrt(6/fan_in).
- `softmax`: disponível somente como última camada de **duas ou mais** saídas, com exponenciação estabilizada pela subtração do logit máximo; para treinar, use `loss='cce'`. MSE não é suportada com softmax.

Não há suporte a redes convolucionais, recorrentes, transformers ou treinamento de modelos já carregados em JS.

## Otimizadores

`train(network, dataset, *, epochs=1000, learning_rate=0.1, seed=0, batch_size=1, optimizer='sgd', momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01, loss='mse')`

- `sgd` (padrão): atualização `w = w - lr * grad`; `batch_size=1` mantém o comportamento legado.
- `momentum`: `v = momentum * v + grad`; `w = w - lr * v`, velocidade inicial zero.
- `adamw`: momentos de primeira e segunda ordem, correção de viés por passo de **mini-batch**, decaimento desacoplado somente para pesos (não vieses). Momentos não são persistidos.
- `batch_size`: inteiro de 1 a 100000; cada batch calcula a **média** dos gradientes, inclusive o último parcial. A amostragem é embaralhada deterministicamente por `seed` a cada época.

Parâmetros inválidos, estados numéricos não finitos ou pesos não exportáveis para float32 causam erro; em caso de erro, os parâmetros do chamador ficam inalterados. Não há checkpoint/resume entre chamadas; não é treinamento online.

## Funções de perda

- `mse` (padrão): `0.5 * sum((prediction-target)^2)` por amostra, retrocompatível.
- `bce`: entropia cruzada binária para **uma saída sigmoid**; aceita alvo entre 0 e 1, calcula a perda pelos logits com softplus estável e derivada `p-y`.
- `cce`: entropia cruzada categórica para **softmax** com 2+ classes; exige vetor de probabilidades não negativas cuja soma é 1 (tolerância 1e-9), calcula perda via log-sum-exp e derivada `p-y`.

`dataset_loss(network, dataset, loss='...')` calcula a mesma função objetivo sem retropropagação. Alvos, perda e arquitetura são validados antes de começar o treino; o teste final usa os pesos completos da chamada.

## Exemplo binário

```python
from training.network import create_network, predict
from training.train import train, dataset_loss
from training.export import save_model

samples = [([-2], [0]), ([-1], [0]), ([1], [1]), ([2], [1])]
model = create_network([1, 4, 1], ['relu', 'sigmoid'], seed=7)
metrics = train(model, samples, epochs=200, learning_rate=0.08,
                optimizer='adamw', batch_size=4, loss='bce', weight_decay=0.0, seed=12)
print(metrics, dataset_loss(model, samples, loss='bce'), predict(model, [2]))
save_model(model, 'modelo.json')
```

O JavaScript continua usando `loadModel` e `predict` sem opções adicionais. `loss`, otimizador, métricas e rótulos pertencem ao treinamento/consumidor, não ao arquivo v1.

## Validação independente e early stopping — P11

`train(..., validation_dataset=None, patience=None, min_delta=0.0, record_history=False)` acrescenta recursos **opt-in**, mantendo retorno e pesos do comportamento antigo quando usados os defaults.

- `validation_dataset` usa o mesmo formato de `dataset`, mas seus exemplos nunca participam dos gradientes.
- A API rejeita atributos de entrada idênticos entre treinamento e validação; a detecção não identifica grupos relacionados ou duplicatas semânticas. O projeto consumidor deve impedir vazamento por usuário, tempo ou origem.
- A perda de validação é avaliada a cada época, incluindo a época **0**, antes de atualizar pesos.
- O melhor checkpoint é guardado quando há redução estrita da perda superior a `min_delta`; ao terminar ou interromper, os melhores pesos são restaurados.
- `patience` é um inteiro positivo que interrompe após essa quantidade de épocas consecutivas sem melhoria; exige `validation_dataset`.
- Com validação, o retorno agrega `initialValidationLoss`, `bestValidationLoss`, `bestEpoch`, `epochsRun` e `stoppedEarly`. `epochs` continua indicando o máximo solicitado.
- `record_history=True` acrescenta o histórico por época de `trainLoss` e, se disponível, `validationLoss`; pode usar memória proporcional ao número de épocas. Não há histórico por padrão.

Exemplo de uso:

```python
metrics = train(model, training_samples, loss="bce", epochs=250,
                validation_dataset=validation_samples, patience=20,
                min_delta=1e-4, record_history=True, optimizer="adamw")
# Só depois que os pesos finais forem selecionados:
# avaliar uma única vez em um conjunto de teste intocado.
```

Os conjuntos de teste não são argumento de `train`. Early stopping não garante superioridade; o melhor modelo depende da distribuição dos dados de validação.

## Verificação e limites

O CI valida derivadas numericamente, atualizações de mini-batches/otimizadores, casos extremos, aprendizagem sintética e paridade de inferência Python × JS para as novas ativações. Os datasets são artificiais: nenhum método é garantidamente melhor em tarefas reais. Para avaliação com conjuntos independentes e baselines, veja [evaluation.md](evaluation.md). 
