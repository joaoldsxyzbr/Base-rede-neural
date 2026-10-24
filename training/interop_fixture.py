"""P14: modelos e descritores Python como fonte independente para CI JS.

python -m training.interop_fixture > caminho.json
Sem download, telemetria, dependências externas ou gravações implícitas.
"""
import json
import sys

from .export import describe_model, export_model, get_capabilities
from .network import create_network


def build_fixture():
    cases = [
        ("relu-softmax", [2, 4, 3], ["relu", "softmax"], 19),
        ("tanh-sigmoid", [3, 2, 1], ["tanh", "sigmoid"], 7),
        ("linear", [1, 1], ["linear"], 11),
    ]
    models = []
    for name, sizes, activations, seed in cases:
        model = export_model(create_network(sizes, activations, seed=seed))
        models.append({
            "name": name,
            "model": model,
            "description": describe_model(model),
        })
    return {
        "conformanceVersion": 1,
        "capabilities": get_capabilities(),
        "models": models,
    }


def main():
    json.dump(build_fixture(), sys.stdout, ensure_ascii=False, allow_nan=False,
              separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
