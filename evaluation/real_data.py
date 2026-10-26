"""P16: benchmark científico mínimo com dados reais, somente biblioteca padrão.

Uso: python -m evaluation.real_data
Os 150 registros Iris são reais, mas não representam aplicações de produção.
"""
import csv
import json
import math
import random
import statistics
from pathlib import Path

from training.network import create_network, predict
from training.train import train

LABELS = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")
DATASET = Path(__file__).resolve().parent / "data" / "iris.csv"


def load_iris(path=None):
    """Carrega cópia local versionada, sem HTTP, pacotes ou dados pessoais."""
    rows = []
    with Path(path or DATASET).open(encoding="utf-8", newline="") as stream:
        for number, fields in enumerate(csv.reader(stream), 1):
            if not fields:
                continue
            if len(fields) != 5 or fields[4] not in LABELS:
                raise ValueError(f"Iris linha {number}: colunas ou rótulo inválidos")
            try:
                features = tuple(float(value) for value in fields[:4])
            except ValueError as exc:
                raise ValueError(f"Iris linha {number}: atributo não numérico") from exc
            if not all(math.isfinite(value) for value in features):
                raise ValueError(f"Iris linha {number}: atributo não finito")
            rows.append((features, LABELS.index(fields[4])))
    if not rows:
        raise ValueError("Dataset Iris vazio")
    return rows


def split_stratified(rows, *, seed):
    """Partições estratificadas por classe; atributos duplicados ficam juntos."""
    if type(seed) is not int or not rows:
        raise ValueError("Seed inteira e registros não vazios obrigatórios")
    grouped = {}
    for features, label in rows:
        if (not isinstance(features, (tuple, list)) or len(features) == 0
                or type(label) is not int or label not in (0, 1, 2)
                or any(not isinstance(x, (int, float)) or isinstance(x, bool)
                       or not math.isfinite(x) for x in features)):
            raise ValueError("Registro inválido")
        key = tuple(features)
        if key in grouped and grouped[key][0][1] != label:
            raise ValueError("Mesmo vetor possui rótulos conflitantes")
        grouped.setdefault(key, []).append((key, label))

    by_class = {label: [] for label in range(len(LABELS))}
    for group in grouped.values():
        by_class[group[0][1]].append(group)

    train_rows, val_rows, test_rows = [], [], []
    for label, groups in by_class.items():
        if len(groups) < 3:
            raise ValueError(f"Classe {label} possui grupos insuficientes para divisão")
        random.Random(seed + 1009 * label).shuffle(groups)
        n_test = max(1, round(len(groups) * 0.2))
        n_val = max(1, round(len(groups) * 0.2))
        if n_test + n_val >= len(groups):
            raise ValueError("Sem grupos suficientes para treino")
        for group in groups[:n_test]:
            test_rows.extend(group)
        for group in groups[n_test:n_test + n_val]:
            val_rows.extend(group)
        for group in groups[n_test + n_val:]:
            train_rows.extend(group)

    for offset, part in enumerate((train_rows, val_rows, test_rows)):
        random.Random(seed + 101 + offset).shuffle(part)
    return train_rows, val_rows, test_rows


def standardize(training, validation, test):
    """Calcula médias e desvios APENAS no conjunto de treinamento."""
    if not training:
        raise ValueError("Treinamento vazio")
    width = len(training[0][0])
    if width < 1 or any(len(features) != width for part in (training, validation, test)
                        for features, _ in part):
        raise ValueError("Larguras incompatíveis")
    mean = tuple(sum(features[i] for features, _ in training) / len(training)
                 for i in range(width))
    scale = tuple(math.sqrt(sum((features[i] - mean[i]) ** 2
                                 for features, _ in training) / len(training)) or 1.0
                  for i in range(width))

    def transform(part):
        result = []
        for features, label in part:
            vector = tuple((features[i] - mean[i]) / scale[i] for i in range(width))
            if not all(math.isfinite(x) for x in vector):
                raise ValueError("Normalização produziu número não finito")
            result.append((vector, label))
        return result

    return transform(training), transform(validation), transform(test), mean, scale


def multiclass_metrics(actual, predicted, *, class_count):
    """Acurácia, macro-F1 e matriz: linhas classe real, colunas prevista."""
    if (type(class_count) is not int or class_count < 2 or len(actual) != len(predicted)
            or not actual):
        raise ValueError("Rótulos ou número de classes inválidos")
    matrix = [[0] * class_count for _ in range(class_count)]
    for y, p in zip(actual, predicted):
        if (type(y) is not int or type(p) is not int
                or not 0 <= y < class_count or not 0 <= p < class_count):
            raise ValueError("Classe fora do intervalo")
        matrix[y][p] += 1
    f1 = []
    for i in range(class_count):
        tp = matrix[i][i]
        fp = sum(row[i] for row in matrix) - tp
        fn = sum(matrix[i]) - tp
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {
        "accuracy": sum(matrix[i][i] for i in range(class_count)) / len(actual),
        "macroF1": statistics.mean(f1),
        "confusion": matrix,
    }


def _centroid_predictor(training):
    centers = []
    for label in range(len(LABELS)):
        vectors = [x for x, y in training if y == label]
        if not vectors:
            raise ValueError("Classe ausente no conjunto de treino")
        centers.append(tuple(sum(x[i] for x in vectors) / len(vectors)
                             for i in range(len(vectors[0]))))

    def classify(features):
        return min(range(len(LABELS)),
                   key=lambda label: sum((x - c) ** 2
                                         for x, c in zip(features, centers[label])))
    return classify


def _score(samples, classifier):
    return multiclass_metrics(
        [label for _, label in samples],
        [classifier(features) for features, _ in samples],
        class_count=len(LABELS),
    )


def run_experiment(*, seeds=(7, 19, 31), epochs=160, patience=20):
    """Treina somente no treino; checkpoint por validação; teste apenas ao final."""
    if (not isinstance(seeds, (list, tuple)) or not seeds or len(seeds) > 20
            or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Sementes precisam ser inteiras, distintas e limitadas")
    if (type(epochs) is not int or not 1 <= epochs <= 10000 or
            type(patience) is not int or not 1 <= patience <= 10000):
        raise ValueError("Épocas ou paciência inválidas")

    rows = load_iris()
    runs = []
    for seed in seeds:
        training, validation, heldout = split_stratified(rows, seed=seed)
        training, validation, heldout, means, scales = standardize(
            training, validation, heldout)
        network = create_network([4, 8, 3], ["tanh", "softmax"], seed=seed)
        onehot = lambda samples: [(x, [float(j == y) for j in range(3)])
                                  for x, y in samples]
        info = train(network, onehot(training), epochs=epochs,
                     optimizer="adamw", learning_rate=0.04, batch_size=12,
                     weight_decay=0.0, loss="cce", seed=seed + 4000,
                     validation_dataset=onehot(validation), patience=patience)

        def neural(features):
            probabilities = predict(network, features)
            return max(range(3), key=lambda index: probabilities[index])

        centroids = _centroid_predictor(training)
        counts = [sum(y == label for _, y in training) for label in range(3)]
        majority = max(range(3), key=lambda label: counts[label])
        models = {
            "neuralCore": {"test": _score(heldout, neural)},
            "nearestCentroid": {"test": _score(heldout, centroids)},
            "majority": {"test": _score(heldout, lambda features: majority)},
        }
        runs.append({
            "seed": seed,
            "counts": {"train": len(training), "validation": len(validation),
                       "test": len(heldout)},
            "normalization": {"trainingMean": means, "trainingScale": scales},
            "models": models,
            "training": {"bestEpoch": info["bestEpoch"],
                         "epochsRun": info["epochsRun"],
                         "stoppedEarly": info["stoppedEarly"]},
        })

    summary = {}
    for model in ("neuralCore", "nearestCentroid", "majority"):
        summary[model] = {}
        for metric in ("accuracy", "macroF1"):
            values = [run["models"][model]["test"][metric] for run in runs]
            summary[model][metric + "Mean"] = statistics.mean(values)
            summary[model][metric + "Std"] = statistics.pstdev(values)

    return {
        "protocolVersion": 3,
        "dataset": {"name": "Iris", "instances": len(rows), "features": 4,
                    "classes": list(LABELS), "source": "UCI dataset 53",
                    "license": "CC BY 4.0"},
        "seeds": list(seeds),
        "configuration": {
            "model": [4, 8, 3], "activations": ["tanh", "softmax"],
            "optimizer": "adamw", "loss": "cce", "batchSize": 12,
            "learningRate": 0.04, "weightDecay": 0.0,
            "epochs": epochs, "patience": patience,
            "split": "stratified grouped by identical features (~60/20/20)",
            "normalization": "train only",
            "testSelection": "none; parameters fixed before evaluation",
        },
        "runs": runs,
        "summary": summary,
        "limitations": "Iris é pequeno e simples; dados reais não equivalem a generalização de produção.",
    }


def main():
    print(json.dumps(run_experiment(), ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
