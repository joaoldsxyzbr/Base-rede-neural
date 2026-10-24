"""P14: contrato comum, versões explícitas e descritores portáteis."""
import copy
import json
import unittest

from training.export import describe_model, export_model, get_capabilities
from training.interop_fixture import build_fixture
from training.network import create_network


class InteroperabilityTests(unittest.TestCase):
    def test_capacidades_reais_sem_mutacao_entre_chamadas(self):
        expected = {
            "formatVersions": [1], "operators": ["dense"],
            "activations": ["linear", "relu", "sigmoid", "softmax", "tanh"],
            "limits": {"maxLayers": 16, "maxWidth": 4096, "maxParameters": 250000},
        }
        self.assertEqual(get_capabilities(), expected)
        copy_of_capabilities = get_capabilities()
        copy_of_capabilities["activations"].append("eval")
        copy_of_capabilities["limits"]["maxLayers"] = 999
        self.assertEqual(get_capabilities(), expected)

    def test_descricao_independente_dos_pesos(self):
        model = create_network([2, 3, 2], ["relu", "softmax"], seed=6)
        snapshot = copy.deepcopy(model)
        info = describe_model(model)
        self.assertEqual(info, {
            "formatVersion":1, "inputSize":2, "outputSize":2, "layerCount":2,
            "parameterCount":17,
            "layers":[
                {"inputSize":2,"outputSize":3,"activation":"relu","parameterCount":9},
                {"inputSize":3,"outputSize":2,"activation":"softmax","parameterCount":8},
            ]
        })
        self.assertEqual(model, snapshot)
        self.assertNotIn("weights", json.dumps(info))
        self.assertEqual(describe_model(export_model(model)), info)

    def test_descritor_rejeita_v2_sem_conversao_silenciosa(self):
        model = create_network([1, 1], ["linear"], seed=2)
        for invalid in [None, {}, {**model, "formatVersion": 2}]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                describe_model(invalid)

    def test_fixture_python_deterministica_e_paridade_de_metadados(self):
        a = build_fixture()
        b = build_fixture()
        self.assertEqual(a, b)
        self.assertEqual(a["conformanceVersion"],1)
        self.assertGreaterEqual(len(a["models"]),3)
        self.assertEqual(a["capabilities"],get_capabilities())
        for item in a["models"]:
            self.assertEqual(item["description"],describe_model(item["model"]))
            self.assertEqual(item["model"],export_model(item["model"]))
        json.dumps(a, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
