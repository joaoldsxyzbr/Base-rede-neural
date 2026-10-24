"""Valida que as referências são provenientes de modelos exportados de Python."""
import json
import math
import unittest

from training.network import predict
from training.parity_fixture import build_fixture


class ParityFixtureTests(unittest.TestCase):
    def test_geracao_reproduzivel_sem_dependencias_externas(self):
        first = build_fixture()
        second = build_fixture()
        self.assertEqual(first, second)
        self.assertEqual(first["fixtureVersion"], 1)
        self.assertGreaterEqual(len(first["cases"]), 4)
        json.dumps(first, allow_nan=False)

    def test_previsoes_correspondem_aos_pesos_exportados(self):
        fixture = build_fixture()
        activations = set()
        for case in fixture["cases"]:
            model = case["model"]
            self.assertEqual(model["formatVersion"], 1)
            activations.update(layer["activation"] for layer in model["layers"])
            for sample in case["samples"]:
                reference = predict(model, sample["input"])
                self.assertEqual(reference, sample["expected"])
                self.assertTrue(all(math.isfinite(v) for v in reference))
        self.assertEqual(activations, {"linear", "tanh", "sigmoid", "relu", "softmax"})

    def test_rede_treinada_realmente_aprende_xor(self):
        trained = next(c for c in build_fixture()["cases"] if c["name"] == "xor-trained")
        for sample in trained["samples"][:4]:
            self.assertEqual(sample["expected"][0] >= .5, sample["input"][0] != sample["input"][1])


if __name__ == "__main__":
    unittest.main()
