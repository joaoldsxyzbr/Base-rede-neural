"""P10: algoritmos de treinamento, gradientes, limites e compatibilidade."""
import copy
import math
import unittest

from training.export import export_model
from training.network import create_network, loss_and_gradients, predict
from training.train import dataset_loss, train


class AdvancedTrainingTests(unittest.TestCase):
    def test_relu_he_reproduzivel_e_exportavel(self):
        a = create_network([3, 6, 2], ["relu", "softmax"], seed=12)
        b = create_network([3, 6, 2], ["relu", "softmax"], seed=12)
        self.assertEqual(a, b)
        self.assertTrue(all(abs(w) <= math.sqrt(6 / 3) for w in a["layers"][0]["weights"]))
        self.assertEqual([layer["activation"] for layer in export_model(a)["layers"]], ["relu", "softmax"])
        self.assertEqual(predict(a, [0, 0, 0]), [0.5, 0.5])
        with self.assertRaises(ValueError):
            create_network([2, 3, 2], ["softmax", "softmax"], seed=1)

    def test_derivadas_relu_bce_e_cce_batem_diferencas_finitas(self):
        cases = [
            (["relu", "sigmoid"], "bce", [0.2], [0.8, 0.6], [0.7]),
            (["relu", "softmax"], "cce", [0.0, 1.0], [0.8, 0.6], [0.0, 1.0]),
        ]
        for activations, loss, _, inputs, target in cases:
            with self.subTest(loss=loss):
                model = create_network([2, 3, len(target)], activations, seed=13)
                model["layers"][0]["biases"] = [0.2, 0.3, 0.4]
                _, gradients = loss_and_gradients(model, inputs, target, loss=loss)
                for layer, grad in zip(model["layers"], gradients):
                    for field in ("weights", "biases"):
                        for idx in range(len(layer[field])):
                            original = layer[field][idx]
                            eps = 1e-5
                            layer[field][idx] = original + eps
                            plus = loss_and_gradients(model, inputs, target, loss=loss)[0]
                            layer[field][idx] = original - eps
                            minus = loss_and_gradients(model, inputs, target, loss=loss)[0]
                            layer[field][idx] = original
                            self.assertTrue(math.isclose(grad[field][idx], (plus-minus)/(2*eps),
                                rel_tol=1e-4, abs_tol=1e-6), (loss, field, idx, grad[field][idx]))

    def test_mini_batch_usa_media_dos_gradientes(self):
        model = create_network([1, 1], ["linear"], seed=4)
        model["layers"][0]["weights"][:] = [0.0]
        model["layers"][0]["biases"][:] = [0.0]
        train(model, [([1], [2]), ([2], [4])],
              epochs=1, learning_rate=0.1, seed=2, batch_size=2)
        self.assertAlmostEqual(model["layers"][0]["weights"][0], 0.5)
        self.assertAlmostEqual(model["layers"][0]["biases"][0], 0.3)

    def test_momentum_e_adamw_atualizam_com_formula_correta(self):
        momentum = create_network([1, 1], ["linear"], seed=2)
        momentum["layers"][0]["weights"][:] = [0.0]
        momentum["layers"][0]["biases"][:] = [0.0]
        train(momentum, [([0], [1])], epochs=2, learning_rate=0.1,
              optimizer="momentum", momentum=0.5)
        self.assertAlmostEqual(momentum["layers"][0]["biases"][0], 0.24)

        adam = create_network([1, 1], ["linear"], seed=2)
        adam["layers"][0]["weights"][:] = [1.0]
        adam["layers"][0]["biases"][:] = [0.0]
        train(adam, [([0], [0])], epochs=1, learning_rate=0.1,
              optimizer="adamw", weight_decay=0.1)
        self.assertAlmostEqual(adam["layers"][0]["weights"][0], 0.99)
        self.assertAlmostEqual(adam["layers"][0]["biases"][0], 0.0)

        adam2 = create_network([1, 1], ["linear"], seed=2)
        adam2["layers"][0]["weights"][:] = [0.0]
        adam2["layers"][0]["biases"][:] = [0.0]
        train(adam2, [([1], [1])], epochs=1, learning_rate=0.1,
              optimizer="adamw", weight_decay=0.0)
        self.assertAlmostEqual(adam2["layers"][0]["weights"][0], 0.1, places=6)
        self.assertAlmostEqual(adam2["layers"][0]["biases"][0], 0.1, places=6)

    def test_classificacao_binaria_e_multiclasse_aprendem(self):
        binary = create_network([1, 1], ["sigmoid"], seed=1)
        samples = [([-2], [0]), ([-1], [0]), ([1], [1]), ([2], [1])]
        initial = dataset_loss(binary, samples, loss="bce")
        train(binary, samples, epochs=120, batch_size=4, optimizer="adamw",
              learning_rate=0.08, weight_decay=0.0, loss="bce", seed=7)
        self.assertLess(dataset_loss(binary, samples, loss="bce"), initial * 0.15)
        self.assertGreater(predict(binary, [2])[0], 0.9)

        multi = create_network([3, 3], ["softmax"], seed=1)
        points = [([1, 0, 0], [1, 0, 0]), ([0, 1, 0], [0, 1, 0]), ([0, 0, 1], [0, 0, 1])]
        train(multi, points, epochs=100, batch_size=3, optimizer="adamw",
              learning_rate=0.08, weight_decay=0, loss="cce", seed=7)
        for x, y in points:
            output = predict(multi, x)
            self.assertAlmostEqual(sum(output), 1.0, places=10)
            self.assertGreater(output[y.index(1)], 0.9)

    def test_classification_losses_sao_estaveis_em_extremos(self):
        binary = create_network([1, 1], ["sigmoid"], seed=1)
        binary["layers"][0]["weights"][:] = [1000.0]
        binary["layers"][0]["biases"][:] = [0.0]
        self.assertLess(loss_and_gradients(binary, [1], [1], loss="bce")[0], 1e-8)
        self.assertAlmostEqual(loss_and_gradients(binary, [1], [0], loss="bce")[0], 1000.0)
        multi = create_network([1, 2], ["softmax"], seed=2)
        multi["layers"][0]["weights"][:] = [1000.0, -1000.0]
        multi["layers"][0]["biases"][:] = [0.0, 0.0]
        self.assertEqual(predict(multi, [1]), [1.0, 0.0])
        self.assertAlmostEqual(loss_and_gradients(multi, [1], [0, 1], loss="cce")[0], 2000.0)

    def test_adamw_rejeita_overflow_de_estado_sem_alterar_pesos(self):
        model = create_network([1, 1], ["sigmoid"], seed=2)
        model["layers"][0]["weights"][:] = [1.0]
        previous = copy.deepcopy(model)
        with self.assertRaises(ValueError):
            train(model, [([1e200], [0])], epochs=1,
                  learning_rate=0.1, optimizer="adamw", loss="bce")
        self.assertEqual(model, previous)

    def test_validacao_preserva_modelo_e_parametros_anteriores(self):
        model = create_network([2, 3, 1], ["relu", "sigmoid"], seed=2)
        before = copy.deepcopy(model)
        weight_lists = [layer["weights"] for layer in model["layers"]]
        samples = [([1, 1], [1])]
        cases = [
            {"optimizer": "rmsprop"}, {"batch_size": 0}, {"batch_size": 1.5},
            {"momentum": 1}, {"beta1": 1}, {"beta2": -0.5}, {"epsilon": 0},
            {"weight_decay": -1}, {"loss": "cce"}, {"loss": "bce", "batch_size": -1},
            {"loss": []},
        ]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                train(model, samples, epochs=1, **options)
            self.assertEqual(before, model)
        with self.assertRaises(ValueError):
            train(model, [([1, 1], [2])], loss="bce", epochs=1)
        self.assertEqual(before, model)
        self.assertTrue(all(a is b["weights"] for a, b in zip(weight_lists, model["layers"])))
        softmax = create_network([1, 2], ["softmax"], seed=2)
        with self.assertRaises(ValueError):
            train(softmax, [([1], [0.7, 0.7])], loss="cce", epochs=1)
        with self.assertRaises(ValueError):
            dataset_loss(softmax, [([1], [1, 0])], loss="bce")


if __name__ == "__main__":
    unittest.main()
