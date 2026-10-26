"""P16: regressões para Airfoil Self-Noise, grupos e baselines de regressão."""
import math
import unittest

from evaluation.airfoil import (
    load_airfoil, split_conditions, normalize_splits, ridge_fit, regression_metrics,
    run_airfoil_experiment,
)


class AirfoilEvaluationTests(unittest.TestCase):
    def test_dataset_real_e_divisao_reprodutivel_sem_vazamento_de_condicoes(self):
        rows = load_airfoil()
        self.assertEqual(len(rows), 1503)
        self.assertTrue(all(len(x) == 5 and math.isfinite(y)
                            and all(math.isfinite(v) for v in x) for x, y in rows))
        groups = split_conditions(rows, seed=7)
        self.assertEqual(groups, split_conditions(rows, seed=7))
        self.assertNotEqual(groups, split_conditions(rows, seed=19))
        self.assertEqual(sum(len(part) for part in groups), len(rows))
        signatures = [{x[1:] for x, _ in part} for part in groups]
        self.assertFalse(signatures[0] & signatures[1])
        self.assertFalse(signatures[0] & signatures[2])
        self.assertFalse(signatures[1] & signatures[2])
        self.assertTrue(all(part for part in groups))

    def test_normalizacao_de_atributos_e_alvo_so_utiliza_treino(self):
        train = [((1.0, 3.0), 10.0), ((3.0, 3.0), 20.0)]
        validation = [((1000.0, 3.0), 1000.0)]
        test = [((-1000.0, 3.0), -1000.0)]
        parts, feature_mean, feature_scale, target_mean, target_scale = normalize_splits(
            train, validation, test)
        self.assertEqual(feature_mean, (2.0, 3.0))
        self.assertEqual(feature_scale, (1.0, 1.0))
        self.assertEqual(target_mean, 15.0)
        self.assertEqual(target_scale, 5.0)
        self.assertEqual(parts[0][0], ((-1.0, 0.0), -1.0))
        self.assertEqual(parts[1][0], ((998.0, 0.0), 197.0))
        self.assertEqual(parts[2][0], ((-1002.0, 0.0), -203.0))

    def test_baseline_linear_e_metricas_verificaveis(self):
        rows = [((float(x), float(y)), 2.0 + 3.0 * x - 4.0 * y)
                for x in range(-2, 3) for y in range(-2, 3)]
        predict = ridge_fit(rows, penalty=0.0)
        self.assertAlmostEqual(predict((0.0, 0.0)), 2.0, places=8)
        self.assertAlmostEqual(predict((3.0, -2.0)), 19.0, places=8)
        result = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
        self.assertAlmostEqual(result["mae"], 1 / 3)
        self.assertAlmostEqual(result["rmse"], math.sqrt(1 / 3))
        self.assertAlmostEqual(result["r2"], 0.5)
        with self.assertRaises(ValueError):
            regression_metrics([1.0], [math.nan])
        with self.assertRaises(ValueError):
            split_conditions([((1.0, 2.0, 3.0, 4.0, 5.0), 1.0)], seed=1)

    def test_experimento_reprodutivel_integra_rede_e_baselines(self):
        report = run_airfoil_experiment(seeds=(7,), epochs=2, patience=1)
        self.assertEqual(report["dataset"]["instances"], 1503)
        self.assertEqual(report["dataset"]["features"], 5)
        run = report["runs"][0]
        self.assertEqual(set(run["models"]), {"neuralCore", "ridgeLinear", "trainMean"})
        self.assertEqual(sum(run["counts"].values()), 1503)
        for item in run["models"].values():
            self.assertTrue(math.isfinite(item["test"]["rmse"]))
            self.assertTrue(math.isfinite(item["test"]["mae"]))
            self.assertTrue(math.isfinite(item["test"]["r2"]))
        self.assertEqual(report["seeds"], [7])


if __name__ == "__main__":
    unittest.main()
