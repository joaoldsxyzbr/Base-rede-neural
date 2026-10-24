import tempfile
import json
import unittest
from unittest.mock import patch
from pathlib import Path

from training.network import create_network
from training.export import export_model, save_model


class ExportTests(unittest.TestCase):
    def test_formato_compativel_com_contrato_e_float32(self):
        model = create_network([2, 3, 1], ["tanh", "sigmoid"], seed=8)
        exported = export_model(model)
        self.assertEqual(exported["formatVersion"], 1)
        self.assertEqual(len(exported["layers"]), 2)
        self.assertEqual(exported["layers"][0]["inputSize"], 2)
        self.assertEqual(exported["layers"][0]["outputSize"], 3)
        self.assertEqual(len(exported["layers"][0]["weights"]), 6)
        self.assertEqual(len(exported["layers"][0]["biases"]), 3)
        self.assertNotEqual(model["layers"][0]["weights"][0], exported["layers"][0]["weights"][0])
        exported["layers"][0]["weights"][0] = 100
        self.assertNotEqual(model["layers"][0]["weights"][0], 100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            save_model(model, path)
            actual = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(actual["formatVersion"], 1)
            self.assertEqual(actual["layers"][1]["activation"], "sigmoid")

    def test_rejeita_pesos_invalidos_e_formato_adulterado(self):
        for corrupt in (float('inf'), float('nan'), 1e100, "1"):
            model = create_network([1, 1], ["linear"])
            model["layers"][0]["weights"][0] = corrupt
            with self.subTest(corrupt=corrupt), self.assertRaises((ValueError, TypeError)):
                export_model(model)
        model = create_network([1, 1], ["linear"])
        model["formatVersion"] = 42
        with self.assertRaises(ValueError):
            export_model(model)
        model = create_network([1, 1], ["linear"])
        model["layers"][0]["weights"].append(0.0)
        with self.assertRaises(ValueError):
            export_model(model)

    def test_save_model_preserves_existing_file_on_replace_failure(self):
        """Falha na substituicao nao pode destruir um modelo anterior valido."""
        model = create_network([1, 1], ["linear"], seed=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            save_model(model, path)
            previous = path.read_text(encoding="utf-8")
            self.assertEqual(json.loads(previous)["formatVersion"], 1)
            model["layers"][0]["biases"][0] += 0.5

            with patch("os.replace", side_effect=OSError("substituicao falhou")):
                with self.assertRaises(OSError):
                    save_model(model, path)

            self.assertEqual(path.read_text(encoding="utf-8"), previous)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["model.json"])

    def test_export_accepts_float32_limits_and_rejects_overflow(self):
        model = create_network([1, 1], ["linear"], seed=2)
        for value in (3.4028234663852886e38, -3.4028234663852886e38, 1e-50):
            with self.subTest(value=value):
                model["layers"][0]["biases"][0] = value
                self.assertEqual(export_model(model)["layers"][0]["biases"][0] == 0.0, value == 1e-50)

        for value in (3.403e38, -3.403e38):
            with self.subTest(value=value):
                model["layers"][0]["biases"][0] = value
                with self.assertRaises(ValueError):
                    export_model(model)


if __name__ == "__main__":
    unittest.main()
