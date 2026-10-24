"""P11: avaliação reproduzível com seleção por validação e teste intocado.

Execute: python -m evaluation.quality
Os dados são sintéticos; resultados não comprovam desempenho em aplicações reais.
"""
import json
import math
import statistics

from .benchmark import binary_metrics, generate_quadrant_samples
from training.network import create_network, predict
from training.train import train


def classification_metrics(targets, probabilities):
    """Acurácia, precisão, recall, F1, Brier e matriz de confusão binária."""
    base = binary_metrics(targets, probabilities)
    tp = tn = fp = fn = 0
    for y, p in zip(targets, probabilities):
        positive = p >= 0.5
        if y == 1 and positive:
            tp += 1
        elif y == 0 and not positive:
            tn += 1
        elif y == 0 and positive:
            fp += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**base, "precision": precision, "recall": recall, "f1": f1,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}


def split_quadrants(*, seed, train_count, validation_count, test_count):
    """Três grupos distintos, balanceados e reprodutíveis (sem reuso de observações)."""
    if type(seed) is not int:
        raise ValueError("seed deve ser inteiro")
    # A função original valida contagens balanceadas e limites.
    train_set = generate_quadrant_samples(train_count, seed=1000 + seed, noise=0.05)
    validation = generate_quadrant_samples(validation_count, seed=100000 + seed, noise=0.05)
    test_set = generate_quadrant_samples(test_count, seed=200000 + seed, noise=0.15)
    groups = [{tuple(x) for x, _ in rows} for rows in (train_set, validation, test_set)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Entradas duplicadas entre treino, validação e teste")
    return train_set, validation, test_set


def _scores(samples, probability):
    return classification_metrics(
        [int(y[0]) for _, y in samples],
        [probability(x) for x, _ in samples],
    )


def _summaries(runs):
    """Médias e dispersões descritivas, não intervalos de confiança."""
    results = {}
    for name in ("sgd", "momentum", "adamw", "linear", "majority"):
        summaries = {}
        for split in ("train", "validation", "test"):
            for metric in ("accuracy", "precision", "recall", "f1", "brier"):
                values = [run["models"][name][split][metric] for run in runs]
                prefix = split + metric[0].upper() + metric[1:]
                summaries[prefix + "Mean"] = statistics.mean(values)
                summaries[prefix + "Std"] = statistics.pstdev(values)
        results[name] = summaries
    return results


def run_quality_experiment(*, seeds=(7, 19, 31), train_count=96,
                           validation_count=64, test_count=128,
                           epochs=160, patience=20):
    """Compara três otimizadores MLP e duas referências, sem seleção no teste.

    Os hiperparâmetros são fixos antecipadamente. Os dados de teste entram
    apenas em _scores, após a conclusão do treinamento de cada modelo.
    """
    if (not isinstance(seeds, (tuple, list)) or not seeds or len(seeds) > 20
            or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds devem conter entre 1 e 20 inteiros distintos")
    if type(epochs) is not int or not 1 <= epochs <= 10_000:
        raise ValueError("epochs fora do intervalo")
    if type(patience) is not int or not 1 <= patience <= 10_000:
        raise ValueError("patience fora do intervalo")
    for name, count in (("train_count", train_count),
                        ("validation_count", validation_count),
                        ("test_count", test_count)):
        if type(count) is not int or not 8 <= count <= 100_000 or count % 4 != 0:
            raise ValueError(f"{name}: inteiro múltiplo de 4 entre 8 e 100000")

    settings = {
        "sgd": {"optimizer": "sgd", "learning_rate": 0.15, "batch_size": 8},
        "momentum": {"optimizer": "momentum", "learning_rate": 0.09, "momentum": 0.8, "batch_size": 8},
        "adamw": {"optimizer": "adamw", "learning_rate": 0.045, "weight_decay": 0.0, "batch_size": 8},
        "linear": {"optimizer": "sgd", "learning_rate": 0.12, "batch_size": 8},
    }
    runs = []
    for seed in seeds:
        training, validation, heldout = split_quadrants(
            seed=seed, train_count=train_count,
            validation_count=validation_count, test_count=test_count)
        models = {}
        for name, options in settings.items():
            model = (create_network([2, 1], ["sigmoid"], seed=seed)
                     if name == "linear" else
                     create_network([2, 8, 1], ["tanh", "sigmoid"], seed=seed))
            stats = train(model, training, epochs=epochs, patience=patience,
                          validation_dataset=validation, min_delta=0.0,
                          loss="bce", seed=seed + 9000, **options)
            probability = lambda x: predict(model, x)[0]
            models[name] = {
                "train": _scores(training, probability),
                "validation": _scores(validation, probability),
                "test": _scores(heldout, probability),
                "training": stats,
            }
        majority_probability = sum(y[0] for _, y in training) / len(training)
        models["majority"] = {
            "train": _scores(training, lambda x: majority_probability),
            "validation": _scores(validation, lambda x: majority_probability),
            "test": _scores(heldout, lambda x: majority_probability),
        }
        runs.append({"seed": seed, "models": models})
    return {
        "protocolVersion": 2,
        "task": "synthetic-quadrants-heldout-validation",
        "seeds": list(seeds),
        "splits": {"train": train_count, "validation": validation_count, "test": test_count},
        "noise": {"trainStd": 0.05, "validationStd": 0.05, "testStd": 0.15},
        "configuration": {"epochs": epochs, "patience": patience,
                          "loss": "bce", "optimizerOptions": settings,
                          "fixedHyperparameters": True, "heldoutUntouched": True},
        "runs": runs,
        "summary": _summaries(runs),
    }


def main():
    print(json.dumps(run_quality_experiment(), ensure_ascii=False,
                     allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
