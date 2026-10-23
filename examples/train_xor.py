"""Exemplo mínimo de treinamento offline do XOR (biblioteca padrão)."""

from training.network import create_network, predict
from training.train import train


if __name__ == "__main__":
    samples = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
    model = create_network([2, 4, 1], ["tanh", "sigmoid"], seed=7)
    result = train(model, samples, epochs=3500, learning_rate=0.35, seed=21)
    print("Treinamento XOR:", result)
    for inputs, target in samples:
        print(f"entrada={inputs}, alvo={target[0]}, previsão={predict(model, inputs)[0]:.5f}")
