import unittest

from training.network import create_network, predict
from training.train import dataset_loss, train


XOR = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]


class LearningTests(unittest.TestCase):
    def test_aprende_xor_com_mesma_seed_e_sem_alterar_dados(self):
        a = create_network([2, 4, 1], ["tanh", "sigmoid"], seed=7)
        b = create_network([2, 4, 1], ["tanh", "sigmoid"], seed=7)
        samples = [(x[:], y[:]) for x, y in XOR]
        result_a = train(a, samples, epochs=3500, learning_rate=0.35, seed=21)
        result_b = train(b, samples, epochs=3500, learning_rate=0.35, seed=21)
        self.assertEqual(a, b)
        self.assertEqual(result_a, result_b)
        self.assertEqual(samples, XOR)
        self.assertGreater(result_a["initialLoss"], 0.08)
        self.assertLess(result_a["finalLoss"], 0.01)
        self.assertLess(result_a["finalLoss"], result_a["initialLoss"] * 0.1)
        for inputs, target in XOR:
            probability = predict(a, inputs)[0]
            self.assertEqual(probability >= 0.5, bool(target[0]))

    def test_aprende_relacao_linear_simples(self):
        samples = [([x], [2*x + 1]) for x in (-2, -1, 0, 1, 2)]
        network = create_network([1, 1], ["linear"], seed=3)
        metrics = train(network, samples, epochs=100, learning_rate=0.05, seed=3)
        self.assertLess(metrics["finalLoss"], 1e-8)
        self.assertLess(dataset_loss(network, samples), metrics["initialLoss"])

    def test_validacao_ocorre_antes_de_qualquer_alteracao_de_pesos(self):
        network = create_network([1, 1], ["linear"])
        previous = repr(network)
        with self.assertRaises(ValueError):
            train(network, [([1], [0]), ([1, 2], [0])], epochs=10)
        self.assertEqual(repr(network), previous)
        for options in ({"epochs": 0}, {"learning_rate": -1}, {"learning_rate": float('nan')}, {"seed": 2.5}):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                train(network, [([1], [0])], **options)
        with self.assertRaises(ValueError):
            train(network, [])


if __name__ == "__main__":
    unittest.main()
