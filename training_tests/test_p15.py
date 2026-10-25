"""P15: testes de contrato reservado, stress de exportação e rollback."""
import copy
import random
import unittest

from training.export import export_model
from training.network import create_network
from training.train import train


class P15HardeningTests(unittest.TestCase):
    def test_operador_dense_compativel_e_outros_rejeitados(self):
        source=create_network([2,3,2], ["relu","softmax"], seed=7)
        reference=export_model(source)
        enriched=copy.deepcopy(source)
        enriched["layers"][0]["operator"]="dense"
        self.assertEqual(export_model(enriched),reference)
        for operator in ("conv", "attention", "eval", None, 123, ["dense"]):
            corrupt=copy.deepcopy(source)
            corrupt["layers"][0]["operator"]=operator
            with self.subTest(operator=operator), self.assertRaises(ValueError):
                export_model(corrupt)
        decorated=copy.deepcopy(source)
        decorated["layers"][0]["description"]="metadata"
        self.assertEqual(export_model(decorated),reference)

    def test_modelos_gerados_têm_parametros_consistentes_sem_mudar_originais(self):
        rng=random.Random(1515)
        activations=("linear","relu","tanh","sigmoid")
        for _ in range(100):
            depth=rng.randrange(1,5)
            widths=[rng.randrange(1,9) for _ in range(depth+1)]
            acts=[rng.choice(activations) for _ in range(depth)]
            if widths[-1]>=2 and rng.random()<0.4:
                acts[-1]="softmax"
            model=create_network(widths,acts,seed=rng.randrange(10_000))
            before=copy.deepcopy(model)
            exported=export_model(model)
            self.assertEqual(model,before)
            self.assertEqual(export_model(exported),exported)
            self.assertEqual(sum(len(l["weights"])+len(l["biases"]) for l in exported["layers"]),
                             sum((a+1)*b for a,b in zip(widths,widths[1:])))

    def test_falhas_de_validacao_nao_modificam_pesos_nem_referencias(self):
        net=create_network([2,2,1],["tanh","sigmoid"],seed=15)
        before=copy.deepcopy(net)
        refs=[(l["weights"],l["biases"]) for l in net["layers"]]
        data=[([0,0],[0]),([1,1],[1])]
        invalid=[
            {"learning_rate":float('nan')},
            {"learning_rate":1e150},
            {"validation_dataset":[([0,0],[0])]},
            {"optimizer":"unexpected"},
            {"batch_size":True},
            {"patience":0,"validation_dataset":[([0.5,0.5],[1])]},
        ]
        for opts in invalid:
            with self.subTest(opts=opts),self.assertRaises(ValueError):
                train(net,data,epochs=5,loss="bce",**opts)
            self.assertEqual(net,before)
            for original,layer in zip(refs,net["layers"]):
                self.assertIs(original[0],layer["weights"])
                self.assertIs(original[1],layer["biases"])


if __name__=="__main__":
    unittest.main()
