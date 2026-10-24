"""Gera referências Python reproduzíveis para testes do motor JavaScript.

Uso: python -m training.parity_fixture > tests/fixtures/parity.json
Não instala dependências nem usa rede. Escreve apenas em stdout.
"""
import json
import math
import sys

from .export import export_model, _as_float32
from .network import create_network, predict
from .train import train


def build_fixture():
    xor_data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
    xor = create_network([2, 4, 1], ["tanh", "sigmoid"], seed=7)
    metrics = train(xor, xor_data, epochs=3500, learning_rate=0.35, seed=21)
    if metrics["finalLoss"] >= 0.01:
        raise ValueError("Treinamento XOR não convergiu")

    linear = create_network([3, 2], ["linear"], seed=11)
    mixed = create_network([3, 4, 3, 2], ["tanh", "sigmoid", "linear"], seed=29)
    sigmoid = create_network([1, 1], ["sigmoid"], seed=0)
    sigmoid["layers"][0]["weights"] = [1.0]
    sigmoid["layers"][0]["biases"] = [0.0]

    relu = create_network([2, 3, 2], ["relu", "linear"], seed=15)
    softmax = create_network([2, 3], ["softmax"], seed=9)
    cases = [
        ("xor-trained", xor, [[0, 0], [0, 1], [1, 0], [1, 1], [0.3, 0.8], [-0.2, 1.2]]),
        ("linear-multi-output", linear, [[0, 0, 0], [1.5, -3.2, 0.25], [-1, 1, 2]]),
        ("mixed-activations", mixed, [[0.0, 0.0, 0.0], [0.1, -0.6, 1.3], [-2, 3, -4]]),
        ("stable-sigmoid", sigmoid, [[-1000], [-120], [-0.01], [0], [0.01], [120], [1000]]),
        ("relu-hidden", relu, [[1, -2], [0, 0], [-0.4, 0.3], [3.1, 0.25]]),
        ("softmax-output", softmax, [[0, 0], [1, 2], [-2, 3], [1000, -1000]]),
    ]
    result = []
    for name, trained, inputs in cases:
        model = export_model(trained)  # Referências usam os pesos float32 exportados.
        samples = []
        for raw_input in inputs:
            converted_input = [_as_float32(value) for value in raw_input]
            expected = predict(model, converted_input)
            if not all(math.isfinite(value) for value in expected):
                raise ValueError("Amostra de paridade não finita")
            samples.append({"input": converted_input, "expected": expected})
        result.append({"name": name, "model": model, "samples": samples})
    return {"fixtureVersion": 1, "cases": result}


def main():
    json.dump(build_fixture(), sys.stdout, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
