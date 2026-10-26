"""P16: avaliação offline NASA/UCI Airfoil Self-Noise (regressão).
Execute: python -m evaluation.airfoil. Depende apenas da biblioteca padrão.
"""
import json
import math
import random
import statistics
from pathlib import Path

from training.network import create_network, predict
from training.train import train

DATASET = Path(__file__).resolve().parent / "data" / "airfoil_self_noise.dat"


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def load_airfoil(path=None):
    """Carrega observações reais sem download ou mudança das unidades originais."""
    rows = []
    with Path(path or DATASET).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"Airfoil linha {line_number}: esperado 6 colunas")
            try:
                values = tuple(float(field) for field in fields)
            except ValueError as exc:
                raise ValueError(f"Airfoil linha {line_number}: número inválido") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Airfoil linha {line_number}: número não finito")
            rows.append((values[:5], values[5]))
    if not rows:
        raise ValueError("Airfoil vazio")
    return rows


def _validate_rows(rows):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("Dataset vazio ou inválido")
    dimension = len(rows[0][0])
    if dimension < 2:
        raise ValueError("Esperados pelo menos dois atributos")
    for sample in rows:
        if (not isinstance(sample, (tuple, list)) or len(sample) != 2
                or not isinstance(sample[0], (tuple, list))
                or len(sample[0]) != dimension
                or not all(_finite(x) for x in sample[0])
                or not _finite(sample[1])):
            raise ValueError("Registro de regressão inválido")
    return dimension


def split_conditions(rows, *, seed):
    """Evita que mesma condição física (exceto frequência) vaze entre partições.

    Agrupamento por ângulo, comprimento, velocidade e deslocamento. Frequências
    de uma mesma condição permanecem no mesmo conjunto, mesmo com alvos distintos.
    """
    if type(seed) is not int:
        raise ValueError("A seed deve ser inteira")
    _validate_rows(rows)
    groups = {}
    for features, target in rows:
        key = tuple(features[1:])
        groups.setdefault(key, []).append((tuple(features), target))
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("Pelo menos três condições independentes são necessárias")
    random.Random(seed).shuffle(keys)
    n_test = max(1, round(len(keys) * 0.2))
    n_val = max(1, round(len(keys) * 0.2))
    if n_test + n_val >= len(keys):
        raise ValueError("Condições insuficientes para treino")
    parts = ([], [], [])
    for i, key in enumerate(keys):
        index = 2 if i < n_test else 1 if i < n_test + n_val else 0
        parts[index].extend(groups[key])
    for i, part in enumerate(parts):
        random.Random(seed + 100 + i).shuffle(part)
    return parts


def normalize_splits(training, validation, heldout):
    """Padroniza entradas e alvo usando APENAS estatísticas de treinamento."""
    width = _validate_rows(training)
    for part in (validation, heldout):
        if not part or _validate_rows(part) != width:
            raise ValueError("Conjuntos vazios ou dimensões incompatíveis")
    means = tuple(sum(x[i] for x, _ in training) / len(training)
                  for i in range(width))
    scales = tuple(math.sqrt(sum((x[i] - means[i]) ** 2 for x, _ in training)
                             / len(training)) or 1.0 for i in range(width))
    target_mean = statistics.mean(y for _, y in training)
    target_scale = math.sqrt(sum((y - target_mean) ** 2 for _, y in training)
                             / len(training)) or 1.0
    if not all(map(math.isfinite, (*means, *scales, target_mean, target_scale))):
        raise ValueError("Estatísticas de normalização inválidas")

    def transform(rows):
        result = []
        for x, y in rows:
            scaled = tuple((x[i] - means[i]) / scales[i] for i in range(width))
            target = (y - target_mean) / target_scale
            if not all(map(math.isfinite, (*scaled, target))):
                raise ValueError("Normalização não finita")
            result.append((scaled, target))
        return result

    return (transform(training), transform(validation), transform(heldout)), (
        means), scales, target_mean, target_scale


def _linear_solve(matrix, rhs):
    """Eliminação Gauss-Jordan com pivoteamento parcial; matrizes pequenas."""
    n = len(rhs)
    values = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda i: abs(values[i][column]))
        if abs(values[pivot][column]) < 1e-12:
            raise ValueError("Sistema linear singular")
        values[column], values[pivot] = values[pivot], values[column]
        divisor = values[column][column]
        values[column] = [v / divisor for v in values[column]]
        for i in range(n):
            if i != column:
                factor = values[i][column]
                if factor:
                    values[i] = [v - factor * u
                                 for v, u in zip(values[i], values[column])]
    answer = tuple(row[-1] for row in values)
    if not all(map(math.isfinite, answer)):
        raise ValueError("Regressão produziu coeficientes inválidos")
    return answer


def ridge_fit(training, *, penalty=0.01):
    """Referência linear com penalidade L2 (apenas inclinações) e intercepto."""
    width = _validate_rows(training)
    if not _finite(penalty) or penalty < 0:
        raise ValueError("Penalidade inválida")
    n = width + 1
    gram = [[0.0] * n for _ in range(n)]
    rhs = [0.0] * n
    for features, target in training:
        values = (1.0, *features)
        for i in range(n):
            rhs[i] += values[i] * target
            for j in range(n):
                gram[i][j] += values[i] * values[j]
    for i in range(1, n):
        gram[i][i] += penalty * len(training)
    beta = _linear_solve(gram, rhs)

    def estimate(features):
        if len(features) != width or not all(_finite(v) for v in features):
            raise ValueError("Atributos inválidos na regressão")
        value = beta[0] + sum(b * x for b, x in zip(beta[1:], features))
        if not math.isfinite(value):
            raise ValueError("Predição linear não finita")
        return value

    return estimate


def regression_metrics(actual, predicted):
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Listas de métricas vazias ou incompatíveis")
    if not all(_finite(y) for y in (*actual, *predicted)):
        raise ValueError("Métricas exigem números finitos")
    errors = [(y - p) ** 2 for y, p in zip(actual, predicted)]
    mean = statistics.mean(actual)
    total = sum((y - mean) ** 2 for y in actual)
    squared = sum(errors)
    values = {
        "rmse": math.sqrt(squared / len(actual)),
        "mae": sum(abs(y - p) for y, p in zip(actual, predicted)) / len(actual),
        "r2": 1 - squared / total if total else (1.0 if squared == 0 else 0.0),
    }
    if not all(map(math.isfinite, values.values())):
        raise ValueError("Resultado de métrica não finito")
    return values


def _scores(rows, estimate):
    return regression_metrics([y for _, y in rows],
                              [estimate(features) for features, _ in rows])


def run_airfoil_experiment(*, seeds=(7, 19, 31), epochs=80, patience=12):
    """Treina por condições disjuntas, seleciona na validação, avalia no teste."""
    if (not isinstance(seeds, (list, tuple)) or not seeds or len(seeds) > 20
            or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Sementes precisam ser inteiras, distintas e limitadas")
    if (type(epochs) is not int or not 1 <= epochs <= 500
            or type(patience) is not int or not 1 <= patience <= 500):
        raise ValueError("Épocas ou paciência inválidas")

    rows = load_airfoil()
    runs = []
    for seed in seeds:
        original = split_conditions(rows, seed=seed)
        (training, validation, heldout), means, scales, y_mean, y_scale = (
            normalize_splits(*original))
        model = create_network([5, 16, 1], ["tanh", "linear"], seed=seed)
        outcome = train(
            model, [(x, [y]) for x, y in training],
            validation_dataset=[(x, [y]) for x, y in validation],
            optimizer="adamw", learning_rate=0.012, batch_size=24,
            weight_decay=0.0, loss="mse",
            epochs=epochs, patience=patience, seed=seed + 8000)

        linear = ridge_fit(training, penalty=0.01)

        def restore(prediction):
            return prediction * y_scale + y_mean

        predictors = {
            "neuralCore": lambda x: restore(predict(model, x)[0]),
            "ridgeLinear": lambda x: restore(linear(x)),
            "trainMean": lambda x: y_mean,
        }
        runs.append({
            "seed": seed,
            "counts": {"train": len(training), "validation": len(validation),
                       "test": len(heldout)},
            "conditionGroups": {
                "train": len({x[1:] for x, _ in original[0]}),
                "validation": len({x[1:] for x, _ in original[1]}),
                "test": len({x[1:] for x, _ in original[2]}),
            },
            "training": {
                "bestEpoch": outcome["bestEpoch"],
                "epochsRun": outcome["epochsRun"],
                "stoppedEarly": outcome["stoppedEarly"],
            },
            "models": {
                name: {"test": regression_metrics(
                    [y for _, y in original[2]],
                    [estimate(x) for x, _ in heldout])}
                for name, estimate in predictors.items()
            },
        })

    summary = {}
    for name in ("neuralCore", "ridgeLinear", "trainMean"):
        summary[name] = {}
        for metric in ("rmse", "mae", "r2"):
            values = [run["models"][name]["test"][metric] for run in runs]
            summary[name][metric + "Mean"] = statistics.mean(values)
            summary[name][metric + "Std"] = statistics.pstdev(values)

    return {
        "protocolVersion": 4,
        "dataset": {"name": "Airfoil Self-Noise", "instances": len(rows),
                    "features": 5, "target": "scaled sound pressure (dB)",
                    "source": "NASA / UCI dataset 291", "license": "CC BY 4.0"},
        "seeds": list(seeds),
        "configuration": {
            "split": "physical conditions grouped by angle,chord,speed,displacement",
            "scaling": "features and targets train only",
            "network": [5, 16, 1], "activations": ["tanh", "linear"],
            "optimizer": "adamw", "loss": "mse",
            "learningRate": 0.012, "batchSize": 24, "weightDecay": 0.0,
            "epochs": epochs, "patience": patience,
            "ridgePenalty": 0.01,
            "hyperparametersFixed": True, "testUsedForSelection": False,
        },
        "runs": runs, "summary": summary,
        "limitations": "Condições do túnel distintas no mesmo dataset; não prova extrapolação para novos equipamentos ou produção.",
    }


def main():
    print(json.dumps(run_airfoil_experiment(), ensure_ascii=False,
                     allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
