import math
import unittest

from training.network import create_network, loss_and_gradients


class GradientTests(unittest.TestCase):
    def test_derivadas_analiticas_batem_com_diferencas_finitas(self):
        # Duas camadas, pesos e vieses, com todas as ativações usadas na v0.1.
        for output_activation in ("linear", "sigmoid", "tanh"):
            with self.subTest(output_activation=output_activation):
                network = create_network([2, 3, 2], ["tanh", output_activation], seed=17)
                x, target = [0.3, -0.7], [0.4, 0.8]
                _, gradients = loss_and_gradients(network, x, target)
                epsilon = 1e-5
                for layer, gradient in zip(network["layers"], gradients):
                    for field in ("weights", "biases"):
                        for index in range(len(layer[field])):
                            original = layer[field][index]
                            layer[field][index] = original + epsilon
                            plus, _ = loss_and_gradients(network, x, target)
                            layer[field][index] = original - epsilon
                            minus, _ = loss_and_gradients(network, x, target)
                            layer[field][index] = original
                            observed = (plus - minus) / (2 * epsilon)
                            self.assertTrue(math.isclose(
                                observed, gradient[field][index], rel_tol=1e-4, abs_tol=1e-6
                            ), (field, index, observed, gradient[field][index]))

    def test_rejeita_dimensoes_e_numeros_invalidos(self):
        for sizes in ([2], [2, 0, 1], [2, 4097, 1], [2, 3.5, 1], [2, 3] * 10):
            with self.subTest(sizes=sizes), self.assertRaises((ValueError, TypeError)):
                create_network(sizes, ["tanh"] * (len(sizes) - 1), seed=1)
        with self.assertRaises(ValueError):
            create_network([2, 1], ["unknown"])
        with self.assertRaises(ValueError):
            create_network([2, 1], [])
        network = create_network([2, 1], ["linear"])
        for inputs, target in (([0], [1]), ([0, float('nan')], [1]), ([0, 1], [float('inf')])):
            with self.subTest(inputs=inputs, target=target), self.assertRaises((ValueError, TypeError)):
                loss_and_gradients(network, inputs, target)


if __name__ == "__main__":
    unittest.main()
