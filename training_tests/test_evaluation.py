"""Scientific quality gates for deterministic held-out experiments."""
import json
import math
import unittest

from evaluation.benchmark import (
    binary_metrics, generate_quadrant_samples, run_experiment,
    regression_metrics, run_regression_experiment,
)


class ScientificEvaluationTests(unittest.TestCase):
    def test_balanced_deterministic_nonoverlapping_splits(self):
        train = generate_quadrant_samples(96, seed=101, noise=0)
        test = generate_quadrant_samples(128, seed=2101, noise=0)
        self.assertEqual(train, generate_quadrant_samples(96, seed=101, noise=0))
        self.assertEqual(sum(target[0] for _, target in train), 48)
        self.assertEqual(sum(target[0] for _, target in test), 64)
        train_features = {tuple(x) for x, _ in train}
        test_features = {tuple(x) for x, _ in test}
        self.assertFalse(train_features.intersection(test_features))
        self.assertNotEqual(train, test)

    def test_noise_is_reproducible_and_does_not_change_labels(self):
        clean = generate_quadrant_samples(64, seed=11, noise=0)
        noisy = generate_quadrant_samples(64, seed=11, noise=0.15)
        self.assertEqual([label for _, label in clean], [label for _, label in noisy])
        self.assertNotEqual([x for x, _ in clean], [x for x, _ in noisy])
        self.assertEqual(noisy, generate_quadrant_samples(64, seed=11, noise=0.15))

    def test_metrics_validate_predictions(self):
        result = binary_metrics([0, 1, 0, 1], [0.1, 0.8, 0.4, 0.6])
        self.assertEqual(result['accuracy'], 1)
        self.assertTrue(math.isclose(result['brier'], (0.01 + 0.04 + 0.16 + 0.16) / 4))
        for expected, predicted in (([], []), ([0], [float('nan')]), ([0], [1.1]), ([2], [0.3]), ([0, 1], [0.5])):
            with self.subTest(expected=expected, predicted=predicted), self.assertRaises(ValueError):
                binary_metrics(expected, predicted)

    def test_experiment_is_reproducible_with_heldout_data_and_baselines(self):
        first = run_experiment(seeds=(7, 19), train_count=96, test_count=128, epochs=350)
        second = run_experiment(seeds=(7, 19), train_count=96, test_count=128, epochs=350)
        self.assertEqual(first, second)
        self.assertEqual(first['protocolVersion'], 1)
        self.assertEqual(first['data']['trainCount'], 96)
        self.assertEqual(first['data']['testCount'], 128)
        self.assertEqual(len(first['runs']), 2)
        json.dumps(first, allow_nan=False)
        for result in first['runs']:
            for name in ('mlp', 'linear', 'majority'):
                self.assertGreaterEqual(result[name]['test']['accuracy'], 0)
                self.assertLessEqual(result[name]['test']['accuracy'], 1)
                self.assertGreaterEqual(result[name]['noisy']['brier'], 0)
            self.assertEqual(result['majority']['test']['accuracy'], 0.5)
            self.assertIn('train', result['mlp'])

    def test_nonlinear_network_generalizes_better_than_linear_and_majority(self):
        result = run_experiment(seeds=(7, 19), train_count=96, test_count=128, epochs=350)
        summary = result['summary']
        self.assertGreater(summary['mlp']['testAccuracyMean'], 0.84)
        self.assertGreater(summary['mlp']['testAccuracyMean'], summary['linear']['testAccuracyMean'] + 0.15)
        self.assertGreater(summary['mlp']['noisyAccuracyMean'], 0.72)
        self.assertGreater(summary['mlp']['testAccuracyMean'], summary['majority']['testAccuracyMean'] + 0.2)

    def test_regression_metrics_validate_and_score_errors(self):
        measured = regression_metrics([0, 1, 2], [0, 1.5, 1])
        self.assertAlmostEqual(measured['mse'], (0 + .25 + 1) / 3)
        self.assertAlmostEqual(measured['mae'], (0 + .5 + 1) / 3)
        for expected, predicted in (([], []), ([1], [float('inf')]), ([0, 1], [2]), ([True], [1])):
            with self.subTest(expected=expected, predicted=predicted), self.assertRaises(ValueError):
                regression_metrics(expected, predicted)

    def test_regression_generalization_matches_least_squares_not_just_mean(self):
        result = run_regression_experiment(seeds=(7, 19), train_count=96, test_count=128)
        self.assertEqual(result, run_regression_experiment(seeds=(7, 19), train_count=96, test_count=128))
        self.assertEqual(result['protocolVersion'], 1)
        self.assertEqual(len(result['runs']), 2)
        self.assertLess(result['summary']['mlp']['testMseMean'], result['summary']['mean']['testMseMean'] * 0.2)
        self.assertLessEqual(result['summary']['mlp']['testMseMean'], result['summary']['leastSquares']['testMseMean'] + .02)
        json.dumps(result, allow_nan=False)

    def test_experiment_rejects_invalid_configuration(self):
        for args in ({'seeds': ()}, {'seeds': (1, 1)}, {'epochs': 0},
                     {'train_count': 3}, {'test_count': 4}, {'train_noise': float('nan')},
                     {'test_noise': -1}, {'learning_rate': 0}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                run_experiment(**args)


if __name__ == '__main__':
    unittest.main()
