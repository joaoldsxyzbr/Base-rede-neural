"""SGD reproduzível em dados locais, sem chamadas externas."""

import math
import random

from .network import _vector, loss_and_gradients


def _samples(network, dataset):
    if not isinstance(dataset, (list, tuple)) or not dataset:
        raise ValueError("Dataset deve conter amostras")
    input_size = network["layers"][0]["inputSize"]
    output_size = network["layers"][-1]["outputSize"]
    samples = []
    for sample in dataset:
        if not isinstance(sample, (list, tuple)) or len(sample) != 2:
            raise ValueError("Amostra deve ser par (entrada, alvo)")
        x = list(_vector(sample[0], input_size, "Entrada"))
        y = list(_vector(sample[1], output_size, "Alvo"))
        samples.append((x, y))
    return samples


def _mean_loss(network, samples):
    result = sum(loss_and_gradients(network, x, y)[0] for x, y in samples) / len(samples)
    if not math.isfinite(result):
        raise ValueError("Perda média não finita")
    return result


def dataset_loss(network, dataset):
    """Média da perda 1/2 * soma dos erros quadráticos por exemplo."""
    return _mean_loss(network, _samples(network, dataset))


def train(network, dataset, *, epochs=1000, learning_rate=0.1, seed=0):
    """Treina in-place por SGD; preserva a ordem original das amostras.

    Retorna somente métricas. Nenhum log, rede ou arquivo é criado.
    """
    if type(epochs) is not int or not 1 <= epochs <= 1_000_000:
        raise ValueError("epochs deve ser inteiro entre 1 e 1000000")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (float, int)) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate deve ser finito e positivo")
    if type(seed) is not int:
        raise ValueError("seed deve ser um inteiro")
    samples = _samples(network, dataset)  # Tudo validado antes da primeira atualização.
    initial_loss = _mean_loss(network, samples)
    rng = random.Random(seed)
    order = list(range(len(samples)))
    for _ in range(epochs):
        rng.shuffle(order)
        for index in order:
            inputs, targets = samples[index]
            _, gradients = loss_and_gradients(network, inputs, targets)
            for layer, gradient in zip(network["layers"], gradients):
                for field in ("weights", "biases"):
                    for j in range(len(layer[field])):
                        next_value = layer[field][j] - learning_rate * gradient[field][j]
                        if not math.isfinite(next_value):
                            raise ValueError("Treinamento divergiu; ajuste a taxa de aprendizado")
                        layer[field][j] = next_value
    return {"initialLoss": initial_loss, "finalLoss": _mean_loss(network, samples), "epochs": epochs}
