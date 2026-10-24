"""P11: métricas, splits, replicação e ausência de uso de test no ajuste."""
import json
import math
import unittest
from unittest.mock import patch

from evaluation.quality import classification_metrics, run_quality_experiment, split_quadrants


class QualityEvaluationTests(unittest.TestCase):
    def test_metricas_classificacao_e_casos_limite(self):
        stats = classification_metrics([0, 0, 1, 1], [0.1, 0.8, 0.6, 0.9])
        self.assertAlmostEqual(stats["accuracy"], .75)
        self.assertAlmostEqual(stats["precision"], 2 / 3)
        self.assertAlmostEqual(stats["recall"], 1.0)
        self.assertAlmostEqual(stats["f1"], .8)
        self.assertEqual(stats["confusion"], {"tp": 2, "tn": 1, "fp": 1, "fn": 0})
        self.assertAlmostEqual(stats["brier"], (.01 + .64 + .16 + .01) / 4)
        self.assertEqual(classification_metrics([0, 0], [0.1, 0.2])["f1"], 0.0)
        for y, p in [([], []), ([0], [float("nan")]), ([0], [1.5]),
                     ([2], [0.2]), ([0], []), ([1], [True])]:
            with self.subTest(y=y, p=p), self.assertRaises(ValueError):
                classification_metrics(y, p)

    def test_splits_sao_disjuntos_e_deterministicos(self):
        first = split_quadrants(seed=7, train_count=32, validation_count=16, test_count=24)
        self.assertEqual(first, split_quadrants(seed=7, train_count=32, validation_count=16, test_count=24))
        train, validation, test = first
        features = [{tuple(x) for x, _ in rows} for rows in first]
        self.assertFalse(features[0] & features[1])
        self.assertFalse(features[0] & features[2])
        self.assertFalse(features[1] & features[2])
        self.assertEqual([len(rows) for rows in first], [32, 16, 24])
        for rows in first:
            self.assertEqual(sum(y[0] for _, y in rows), len(rows) // 2)

    def test_experimento_reproduzivel_com_baselines_sem_promessas_de_superioridade(self):
        settings = dict(seeds=(7, 19), train_count=32, validation_count=16,
                        test_count=24, epochs=35, patience=8)
        first = run_quality_experiment(**settings)
        second = run_quality_experiment(**settings)
        self.assertEqual(first, second)
        self.assertEqual(first["protocolVersion"], 2)
        self.assertEqual(first["splits"], {"train": 32, "validation": 16, "test": 24})
        self.assertEqual(len(first["runs"]), 2)
        self.assertEqual(set(first["summary"]), {"sgd", "momentum", "adamw", "linear", "majority"})
        for run in first["runs"]:
            for name in ("sgd", "momentum", "adamw", "linear", "majority"):
                result = run["models"][name]
                for key in ("train", "validation", "test"):
                    self.assertGreaterEqual(result[key]["accuracy"], 0)
                    self.assertLessEqual(result[key]["accuracy"], 1)
                    self.assertGreaterEqual(result[key]["brier"], 0)
                if name in ("sgd", "momentum", "adamw", "linear"):
                    self.assertLessEqual(result["training"]["epochsRun"], 35)
                    self.assertTrue(0 <= result["training"]["bestEpoch"] <= result["training"]["epochsRun"])
        json.dumps(first, allow_nan=False)

    def test_heldout_nao_e_passado_a_treinamento_nem_selecao(self):
        import evaluation.quality as quality
        called = quality.train
        seen = []

        def track(network, dataset, **kwargs):
            validation = kwargs.get("validation_dataset")
            self.assertIsNotNone(validation)
            seen.append((dataset, validation))
            return called(network, dataset, **kwargs)

        with patch.object(quality, "train", side_effect=track):
            run_quality_experiment(seeds=(7,), train_count=16, validation_count=8,
                                   test_count=12, epochs=6, patience=2)
        self.assertEqual(len(seen), 4)
        held_out = split_quadrants(seed=7, train_count=16, validation_count=8, test_count=12)[2]
        heldout_inputs = {tuple(x) for x, _ in held_out}
        for training, validation in seen:
            self.assertFalse({tuple(x) for x, _ in training} & heldout_inputs)
            self.assertFalse({tuple(x) for x, _ in validation} & heldout_inputs)

    def test_configuracoes_invalidas_sao_rejeitadas(self):
        for opts in ({"seeds": ()}, {"seeds": (7, 7)}, {"train_count": 3},
                     {"validation_count": 0}, {"test_count": 7},
                     {"epochs": 0}, {"patience": 0}, {"seeds": (True,)},
                     {"seeds": tuple(range(25))}):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                run_quality_experiment(**opts)


if __name__ == "__main__":
    unittest.main()
