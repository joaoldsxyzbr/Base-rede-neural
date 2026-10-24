"""P6 regressions: robust training, numerical boundaries and loss evaluation."""
import unittest
from unittest.mock import patch

from training.network import create_network, predict, loss_and_gradients
from training.train import dataset_loss, train
from training.export import export_model


class RobustnessTests(unittest.TestCase):
    def test_failed_training_keeps_entire_model_and_list_references(self):
        network = create_network([1, 1], ["linear"], seed=3)
        previous = repr(network)
        weights = network["layers"][0]["weights"]
        biases = network["layers"][0]["biases"]
        with self.assertRaises(ValueError):
            train(network, [([1], [1])], epochs=5, learning_rate=1e100, seed=3)
        self.assertEqual(repr(network), previous)
        self.assertIs(network["layers"][0]["weights"], weights)
        self.assertIs(network["layers"][0]["biases"], biases)

    def test_multiple_layers_are_unchanged_after_training_diverges(self):
        network = create_network([2, 3, 1], ["tanh", "linear"], seed=13)
        previous = repr(network)
        weights = [layer["weights"] for layer in network["layers"]]
        with self.assertRaises(ValueError):
            train(network, [([1, 1], [1])], epochs=6, learning_rate=1e200, seed=3)
        self.assertEqual(repr(network), previous)
        for layer, original in zip(network["layers"], weights):
            self.assertIs(layer["weights"], original)

    def test_export_rejects_boolean_version(self):
        network = create_network([1, 1], ["linear"])
        network["formatVersion"] = True
        with self.assertRaises(ValueError):
            export_model(network)

    def test_success_updates_existing_parameter_lists_in_place(self):
        network = create_network([1, 1], ["linear"], seed=3)
        weights = network["layers"][0]["weights"]
        biases = network["layers"][0]["biases"]
        old_value = weights[:]
        result = train(network, [([1], [2])], epochs=30, learning_rate=0.1, seed=3)
        self.assertLess(result["finalLoss"], result["initialLoss"])
        self.assertIs(network["layers"][0]["weights"], weights)
        self.assertIs(network["layers"][0]["biases"], biases)
        self.assertNotEqual(weights, old_value)

    def test_evaluation_does_not_compute_gradients(self):
        network = create_network([1, 1], ["linear"], seed=3)
        data = [([0], [1]), ([1], [3]), ([2], [5])]
        with patch('training.train.loss_and_gradients', side_effect=AssertionError('gradient computed')):
            loss = dataset_loss(network, data)
        self.assertGreaterEqual(loss, 0.0)

    def test_very_large_integers_fail_as_controlled_validation_errors(self):
        very_large = 10 ** 400
        network = create_network([1, 1], ["linear"], seed=3)
        cases = (
            lambda: predict(network, [very_large]),
            lambda: loss_and_gradients(network, [0], [very_large]),
            lambda: train(network, [([0], [1])], learning_rate=very_large),
            lambda: export_model({"formatVersion": 1, "layers": [{"inputSize": 1,
                "outputSize": 1, "activation": "linear", "weights": [very_large], "biases": [0]}]})
        )
        for fn in cases:
            with self.subTest(fn=repr(fn)), self.assertRaises(ValueError):
                fn()

    def test_malformed_training_model_fails_before_mutation(self):
        network = create_network([1, 1], ["linear"], seed=3)
        network["layers"][0]["weights"][0] = float('nan')
        previous = repr(network)
        with self.assertRaises(ValueError):
            train(network, [([1], [1])], epochs=3)
        self.assertEqual(repr(network), previous)

    def test_float32_overflow_after_training_does_not_commit(self):
        """A etapa final deve preservar o modelo se os pesos nao couberem em float32."""
        network = create_network([1, 1], ["linear"], seed=3)
        previous = repr(network)
        weights = network["layers"][0]["weights"]
        biases = network["layers"][0]["biases"]

        # SGD termina com perda e parametros finitos em float64, mas bias > float32.
        with self.assertRaisesRegex(ValueError, "float32"):
            train(network, [([0.0], [1e19])], epochs=1, learning_rate=1e20, seed=3)

        self.assertEqual(repr(network), previous)
        self.assertIs(network["layers"][0]["weights"], weights)
        self.assertIs(network["layers"][0]["biases"], biases)


if __name__ == "__main__":
    unittest.main()
