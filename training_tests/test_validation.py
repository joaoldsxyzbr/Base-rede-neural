"""P11: regressões de checkpoint, validação e early stopping."""
import copy
import math
import unittest

from training.network import create_network
from training.train import dataset_loss, train


class ValidationTrainingTests(unittest.TestCase):
    def setUp(self):
        self.samples = [([-2.0], [0.0]), ([-1.0], [0.0]), ([1.0], [1.0]), ([2.0], [1.0])]
        self.validation = [([-1.5], [0.0]), ([1.5], [1.0])]

    def test_sem_validacao_contrato_retorno_legado(self):
        network = create_network([1, 1], ["sigmoid"], seed=7)
        result = train(network, self.samples, epochs=3, seed=8, learning_rate=0.1,
                       optimizer="sgd", loss="bce")
        self.assertEqual(set(result), {"initialLoss", "finalLoss", "epochs"})
        self.assertEqual(result["epochs"], 3)

    def test_parada_antecipada_restaura_melhor_checkpoint(self):
        a = create_network([1, 1], ["sigmoid"], seed=2)
        b = create_network([1, 1], ["sigmoid"], seed=2)
        opts = dict(epochs=60, seed=3, learning_rate=0.05, batch_size=2,
                    optimizer="adamw", weight_decay=0.0, loss="bce",
                    validation_dataset=self.validation, patience=3, min_delta=100.0,
                    record_history=True)
        metrics = train(a, self.samples, **opts)
        self.assertEqual(metrics, train(b, self.samples, **opts))
        self.assertEqual(a, b)
        self.assertTrue(metrics["stoppedEarly"])
        self.assertEqual(metrics["epochsRun"], 3)
        self.assertEqual(metrics["bestEpoch"], 0)
        self.assertEqual(len(metrics["history"]), 4)
        self.assertEqual(metrics["history"][0]["epoch"], 0)
        self.assertAlmostEqual(metrics["bestValidationLoss"],
                               dataset_loss(a, self.validation, loss="bce"), places=12)
        self.assertAlmostEqual(metrics["finalLoss"],
                               dataset_loss(a, self.samples, loss="bce"), places=12)
        self.assertNotEqual(metrics["history"][-1]["validationLoss"], metrics["bestValidationLoss"])

    def test_melhor_epoca_e_restaurada_apos_treino_completo(self):
        net = create_network([1, 1], ["linear"], seed=4)
        train_data = [([0.0], [1.0]), ([1.0], [2.0])]
        val_data = [([0.5], [1.5])]
        metrics = train(net, train_data, validation_dataset=val_data, epochs=25,
                        learning_rate=0.2, record_history=True, seed=5)
        self.assertEqual(metrics["epochsRun"], 25)
        self.assertFalse(metrics["stoppedEarly"])
        self.assertEqual(metrics["bestValidationLoss"], min(x["validationLoss"] for x in metrics["history"]))
        self.assertAlmostEqual(metrics["bestValidationLoss"], dataset_loss(net, val_data))
        self.assertEqual(metrics["history"][metrics["bestEpoch"]]["validationLoss"],
                         metrics["bestValidationLoss"])

    def test_rejeita_vazamento_por_atributos_duplicados_e_dados_invalidos(self):
        for invalid in (
            {"validation_dataset": [([1.0], [0.0])]},
            {"validation_dataset": [([3.0], [1.0])], "patience": 0},
            {"validation_dataset": [([3.0], [1.0])], "patience": 1.5},
            {"validation_dataset": [([3.0], [1.0])], "min_delta": -1},
            {"validation_dataset": [([3.0], [1.0])], "min_delta": float("nan")},
            {"validation_dataset": [([3.0], [1.0])], "min_delta": True},
            {"patience": 2},
            {"record_history": "yes"},
            {"validation_dataset": []},
            {"validation_dataset": [([3.0], [2.0])], "loss": "bce"},
        ):
            net = create_network([1, 1], ["sigmoid"], seed=2)
            original = copy.deepcopy(net)
            lists = [net["layers"][0]["weights"], net["layers"][0]["biases"]]
            options = {"epochs": 4, "learning_rate": 0.05, "loss": "bce", **invalid}
            with self.subTest(options=invalid), self.assertRaises(ValueError):
                train(net, self.samples, **options)
            self.assertEqual(net, original)
            self.assertIs(net["layers"][0]["weights"], lists[0])
            self.assertIs(net["layers"][0]["biases"], lists[1])

    def test_sem_validados_historico_nao_altera_treinamento(self):
        plain = create_network([1, 1], ["linear"], seed=4)
        logged = create_network([1, 1], ["linear"], seed=4)
        a = train(plain, self.samples, epochs=6, seed=2)
        b = train(logged, self.samples, epochs=6, seed=2, record_history=True)
        self.assertEqual(plain, logged)
        self.assertEqual(a, {k: b[k] for k in a})
        self.assertEqual(len(b["history"]), 7)
        self.assertTrue(all("validationLoss" not in item for item in b["history"]))


if __name__ == "__main__":
    unittest.main()
