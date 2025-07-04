import unittest

import numpy as np

from helixnet import *


class SavingTest(unittest.TestCase):
    def test_saving_layer_single_params(self):
        self.maxDiff = None
        dense = layers.Dense(10, 10, lambda x: x, use_bias=False)
        dense.weights.data = np.zeros((10, 10), dtype=np.float32)
        result = models.save_layer(dense)
        expected_no_b = {
            "name": "Dense 1",
            "param_0": np.zeros((10, 10), dtype=np.float32).tolist()
        }
        self.assertDictEqual(result, expected_no_b)

    def test_saving_layer_two_params(self):
        self.maxDiff = None
        dense = layers.Dense(10, 10, lambda x: x, use_bias=True)
        dense.weights.data = np.zeros((10, 10), dtype=np.float32)
        dense.bias.data = np.zeros((1, 10), dtype=np.float32)
        result = models.save_layer(dense)
        expected_no_b = {
            "name": "Dense 1",
            "param_0": np.zeros((10, 10), dtype=np.float32).tolist(),
            "param_1": np.zeros((1, 10), dtype=np.float32).tolist()
        }
        self.assertDictEqual(result, expected_no_b)

    def test_saving_layer_multiple_params(self):
        self.maxDiff = None
