"""Deterministic synthetic held-out experiments and classical baselines."""
import json
import math
import random
import statistics
from training.network import create_network, predict
from training.train import train

def _count(value, label, minimum):
    if type(value) is not int or not minimum <= value <= 100_000 or value % 4 != 0:
        raise ValueError(f"{label}: inteiro múltiplo de quatro, entre {minimum} e 100000")
    return value

def _noise(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label}: ruído inválido")
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}: ruído fora dos limites") from exc
    if not math.isfinite(value) or not 0 <= value <= 0.5:
        raise ValueError(f"{label}: ruído fora dos limites")
    return value

def generate_quadrant_samples(count, *, seed, noise=0.0):
    """Four balanced quadrants. Noise PRNG is independent of latent samples."""
    _count(count, "count", 4)
    if type(seed) is not int:
        raise ValueError("seed: inteiro obrigatório")
    noise = _noise(noise, "noise")
    rng = random.Random(seed)
    jitter = random.Random(seed ^ 0x5DEECE66D)
    samples = []
    for i in range(count):
        x = rng.uniform(0.25, 1.0) * (-1 if i % 2 else 1)
        y = rng.uniform(0.25, 1.0) * (-1 if (i // 2) % 2 else 1)
        label = 1 if (x < 0) != (y < 0) else 0
        if noise:
            x += jitter.gauss(0, noise)
            y += jitter.gauss(0, noise)
        samples.append(([x, y], [label]))
    rng.shuffle(samples)
    return samples

def binary_metrics(targets, probabilities):
    if (not isinstance(targets, (list, tuple)) or
            not isinstance(probabilities, (list, tuple)) or
            not targets or len(targets) != len(probabilities)):
        raise ValueError("Valores esperados e previstos devem ter tamanhos iguais e positivos")
    total = 0.0
    correct = 0
    for expected, predicted in zip(targets, probabilities):
        if type(expected) is not int or expected not in (0, 1):
            raise ValueError("Alvos binários devem ser 0 ou 1")
        if isinstance(predicted, bool) or not isinstance(predicted, (int, float)):
            raise ValueError("Probabilidade inválida")
        try:
            predicted = float(predicted)
        except (OverflowError, ValueError) as exc:
            raise ValueError("Probabilidade inválida") from exc
        if not math.isfinite(predicted) or not 0 <= predicted <= 1:
            raise ValueError("Probabilidade deve estar entre 0 e 1")
        total += (predicted - expected) ** 2
        correct += int((predicted >= 0.5) == bool(expected))
    return {"accuracy": correct / len(targets), "brier": total / len(targets)}

def _measure(samples, probability):
    return binary_metrics([y[0] for _, y in samples],
                          [probability(x) for x, _ in samples])

def run_experiment(*, seeds=(7, 19, 31), train_count=96, test_count=128,
                   epochs=350, learning_rate=0.3, train_noise=0.05, test_noise=0.35):
    """MLP vs linear logistic vs majority; test points never used to fit models."""
    if (not isinstance(seeds, (tuple, list)) or not seeds or
            any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds) or len(seeds) > 20):
        raise ValueError("seeds: inteiros distintos obrigatórios (máximo 20)")
    _count(train_count, "train_count", 8)
    _count(test_count, "test_count", 8)
    if type(epochs) is not int or not 1 <= epochs <= 10000:
        raise ValueError("epochs deve ser inteiro de 1 a 10000")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (float, int)):
        raise ValueError("learning_rate inválida")
    try:
        learning_rate = float(learning_rate)
    except (OverflowError, ValueError) as exc:
        raise ValueError("learning_rate inválida") from exc
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("learning_rate deve estar entre 0 e 1")
    train_noise, test_noise = _noise(train_noise, "train_noise"), _noise(test_noise, "test_noise")
    results = []
    for seed in seeds:
        train_samples = generate_quadrant_samples(train_count, seed=1000 + seed, noise=train_noise)
        held_out = generate_quadrant_samples(test_count, seed=100000 + seed, noise=0)
        held_out_noisy = generate_quadrant_samples(test_count, seed=100000 + seed, noise=test_noise)
        mlp = create_network([2, 8, 1], ["tanh", "sigmoid"], seed=seed)
        linear = create_network([2, 1], ["sigmoid"], seed=seed)
        train(mlp, train_samples, epochs=epochs, learning_rate=learning_rate, seed=seed + 10000)
        train(linear, train_samples, epochs=epochs, learning_rate=learning_rate, seed=seed + 10000)
        majority_probability = sum(y[0] for _, y in train_samples) / len(train_samples)
        run = {"seed": seed}
        for name, probability in (
            ("mlp", lambda x: predict(mlp, x)[0]),
            ("linear", lambda x: predict(linear, x)[0]),
            ("majority", lambda x: majority_probability),
        ):
            run[name] = {
                "train": _measure(train_samples, probability),
                "test": _measure(held_out, probability),
                "noisy": _measure(held_out_noisy, probability),
            }
        results.append(run)
    summary = {}
    for name in ("mlp", "linear", "majority"):
        summary[name] = {}
        for evaluation in ("train", "test", "noisy"):
            for metric in ("accuracy", "brier"):
                vals = [run[name][evaluation][metric] for run in results]
                summary[name][f"{evaluation}{metric.capitalize()}Mean"] = statistics.mean(vals)
                summary[name][f"{evaluation}{metric.capitalize()}Std"] = statistics.pstdev(vals)
    return {"protocolVersion": 1, "task": "synthetic-nonlinear-quadrants",
            "seeds": list(seeds),
            "data": {"trainCount": train_count, "testCount": test_count,
                     "trainNoiseStd": train_noise, "heldoutNoiseStd": test_noise,
                     "minLatentCoordinateMagnitude": 0.25, "disjointSeeds": True},
            "configuration": {"epochs": epochs, "learningRate": learning_rate,
                              "mlp": [2, 8, 1], "linear": [2, 1], "fixedHyperparameters": True},
            "runs": results, "summary": summary}

def regression_metrics(targets, predictions):
    if (not isinstance(targets, (list, tuple)) or
            not isinstance(predictions, (list, tuple)) or
            not targets or len(targets) != len(predictions)):
        raise ValueError("Alvos e previsões devem ter tamanho igual e não vazio")
    errors = []
    for truth, prediction in zip(targets, predictions):
        for value in (truth, prediction):
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError("Regressão exige números finitos")
            try:
                converted = float(value)
            except (ValueError, OverflowError) as exc:
                raise ValueError("Regressão exige números finitos") from exc
            if not math.isfinite(converted):
                raise ValueError("Regressão exige números finitos")
        error = float(prediction) - float(truth)
        if not math.isfinite(error):
            raise ValueError("Erro numérico não finito")
        errors.append(error)
    mse = sum(error * error for error in errors) / len(errors)
    mae = sum(abs(error) for error in errors) / len(errors)
    if not math.isfinite(mse) or not math.isfinite(mae):
        raise ValueError("Erro numérico não finito")
    return {"mse": mse, "mae": mae}

def _regression_data(count, seed, noise):
    rng = random.Random(seed)
    samples = []
    for _ in range(count):
        x = rng.uniform(-1.0, 1.0)
        y = 2.0 * x + 1.0 + rng.gauss(0, noise)
        samples.append(([x], [y]))
    return samples

def _ordinary_least_squares(samples):
    xs = [x[0] for x, _ in samples]
    ys = [y[0] for _, y in samples]
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0:
        raise ValueError("Regressão linear exige variação nas entradas")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    return mean_y - slope * mean_x, slope

def run_regression_experiment(*, seeds=(7, 19, 31), train_count=96, test_count=128):
    """Linear network vs ordinary least squares; no neural superiority expected."""
    if (not isinstance(seeds, (tuple, list)) or not seeds or
            any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds) or len(seeds) > 20):
        raise ValueError("seeds inválidas")
    _count(train_count, "train_count", 8)
    _count(test_count, "test_count", 8)
    runs = []
    for seed in seeds:
        train_set = _regression_data(train_count, 2000 + seed, 0.08)
        held_out = _regression_data(test_count, 200000 + seed, 0.08)
        mlp = create_network([1, 1], ["linear"], seed=seed)
        train(mlp, train_set, epochs=250, learning_rate=0.1, seed=seed + 10000)
        intercept, slope = _ordinary_least_squares(train_set)
        baseline_mean = statistics.mean(y[0] for _, y in train_set)
        run = {"seed": seed}
        for name, func in (
            ("mlp", lambda x: predict(mlp, x)[0]),
            ("leastSquares", lambda x: intercept + slope * x[0]),
            ("mean", lambda x: baseline_mean),
        ):
            run[name] = {}
            for split, examples in (("train", train_set), ("test", held_out)):
                run[name][split] = regression_metrics(
                    [y[0] for _, y in examples], [func(x) for x, _ in examples])
        runs.append(run)
    summary = {}
    for name in ("mlp", "leastSquares", "mean"):
        summary[name] = {}
        for split in ("train", "test"):
            for metric in ("mse", "mae"):
                values = [run[name][split][metric] for run in runs]
                summary[name][f"{split}{metric.capitalize()}Mean"] = statistics.mean(values)
                summary[name][f"{split}{metric.capitalize()}Std"] = statistics.pstdev(values)
    return {"protocolVersion": 1, "task": "synthetic-linear-regression",
            "seeds": list(seeds),
            "data": {"trainCount": train_count, "testCount": test_count,
                     "independentHeldout": True, "noiseStd": 0.08},
            "configuration": {"epochs": 250, "learningRate": 0.1, "mlp": [1, 1]},
            "runs": runs, "summary": summary}

def main():
    print(json.dumps({"quadrants": run_experiment(), "regression": run_regression_experiment()},
                     ensure_ascii=False, allow_nan=False, indent=2))

if __name__ == "__main__":
    main()
