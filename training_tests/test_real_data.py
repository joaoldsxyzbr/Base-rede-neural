"""P16: regressões sobre a avaliação de dados reais e controles de vazamento."""
import math
import unittest

from evaluation.real_data import (
    load_iris, split_stratified, standardize, multiclass_metrics, run_experiment,
)


class RealDataTests(unittest.TestCase):
    def test_iris_real_validado_e_reproduzivel(self):
        rows = load_iris()
        self.assertEqual(len(rows), 150)
        self.assertEqual([sum(label == i for _, label in rows) for i in range(3)], [50, 50, 50])
        self.assertTrue(all(len(features) == 4 and all(math.isfinite(x) for x in features)
                            for features, _ in rows))
        first = split_stratified(rows, seed=7)
        self.assertEqual(first, split_stratified(rows, seed=7))
        self.assertNotEqual(first, split_stratified(rows, seed=19))
        self.assertEqual(sum(map(len, first)), 150)
        sets = [{tuple(x) for x, _ in group} for group in first]
        self.assertFalse((sets[0] & sets[1]) or (sets[0] & sets[2]) or (sets[1] & sets[2]))
        self.assertTrue(all({label for _, label in group} == {0, 1, 2} for group in first))

    def test_duplicatas_de_atributos_nao_vazam_entre_particoes(self):
        rows = [(tuple([float(label), float(i), 0, 0]), label)
                for label in range(3) for i in range(20)]
        rows += [rows[0], rows[0], rows[20], rows[20]]
        train, validation, test = split_stratified(rows, seed=11)
        features = [{x for x, _ in part} for part in (train, validation, test)]
        self.assertFalse(features[0] & features[1])
        self.assertFalse(features[0] & features[2])
        self.assertFalse(features[1] & features[2])
        self.assertEqual(sum(map(len, (train, validation, test))), len(rows))

    def test_normalizacao_utiliza_apenas_treino(self):
        train = [((1.0, 3.0), 0), ((3.0, 3.0), 1)]
        validation = [((1000.0, 3.0), 0)]
        test = [((-1000.0, 3.0), 1)]
        a, b, c, means, scales = standardize(train, validation, test)
        self.assertEqual(means, (2.0, 3.0))
        self.assertEqual(scales, (1.0, 1.0))
        self.assertEqual([x for x, _ in a], [(-1.0, 0.0), (1.0, 0.0)])
        self.assertEqual(b[0][0], (998.0, 0.0))
        self.assertEqual(c[0][0], (-1002.0, 0.0))

    def test_metricas_multiclasse_e_entradas_invalidas(self):
        scores = multiclass_metrics([0, 1, 2], [0, 2, 2], class_count=3)
        self.assertAlmostEqual(scores["accuracy"], 2 / 3)
        self.assertEqual(scores["confusion"], [[1, 0, 0], [0, 0, 1], [0, 0, 1]])
        self.assertAlmostEqual(scores["macroF1"], (1 + 0 + 2 / 3) / 3)
        with self.assertRaises(ValueError):
            multiclass_metrics([0], [3], class_count=3)
        with self.assertRaises(ValueError):
            split_stratified([((1.0, 2.0), 0), ((1.0, 2.0), 1)], seed=1)

    def test_experimento_integrado_retorna_baselines_e_teste(self):
        report = run_experiment(seeds=(7,), epochs=12, patience=4)
        self.assertEqual(report["dataset"]["instances"], 150)
        self.assertEqual(report["seeds"], [7])
        run = report["runs"][0]
        self.assertEqual(set(run["models"]), {"neuralCore", "nearestCentroid", "majority"})
        for model in run["models"].values():
            self.assertTrue(0 <= model["test"]["accuracy"] <= 1)
            self.assertTrue(0 <= model["test"]["macroF1"] <= 1)
            self.assertEqual(sum(map(sum, model["test"]["confusion"])), run["counts"]["test"])


if __name__ == "__main__":
    unittest.main()
