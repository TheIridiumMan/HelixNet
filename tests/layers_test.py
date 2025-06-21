import unittest
from helixnet import layers, loss
import mygrad as mg
import numpy as np


class ABCLayerTest(unittest.TestCase):
    def test_Name_Generation(self):
        case = layers.Layer("D", [])
        self.assertEqual("D", case.name)
        self.assertEqual("D", case.type)
        case = layers.Layer("D", [mg.zeros((2, 3))])
        self.assertEqual("D 1", case.name)
        self.assertEqual("D", case.type)
        case = layers.Layer("D", [mg.zeros((2, 3))])
        self.assertEqual("D 2", case.name)
        self.assertEqual("D", case.type)

    def test_Total_Params(self):
        case = layers.Layer("D", [mg.zeros((2, 3))])
        self.assertEqual(case.total_params(), 6)
        case = layers.Layer("D", [mg.zeros((2, 3)), mg.zeros(3)])
        self.assertEqual(case.total_params(), 9)

    def test_null_grads(self):
        case = layers.Dense(64, 86, lambda x: x)
        # We will be using dense because it is a more complete layer
        loss.MeanAbsError(case.forward(np.ones((1, 64))), np.ones((1, 86)))\
            .backward()
        for param in case.trainable_params:
            self.assertIsNotNone(param.grad)
        case.null_grad()
        for param in case.trainable_params:
            self.assertIsNone(param.grad)

    def test_predict(self):
        case = layers.Dense(64, 86, lambda x: x)
        # We will be using dense because it is a more complete layer
        loss.MeanAbsError(case.forward(np.ones((1, 64))), np.ones((1, 86)))\
            .backward()
        for param in case.trainable_params:
            self.assertIsNone(param.grad)

    def test_output_shape(self):
        case = layers.Conv2D(1, 16, 3, activation=(lambda x: x))
        # Because Convolution uses the inherited method and doesn't provide it's own
        self.assertEqual(case.output_shape([1, 28, 28]), (16, 26, 26))



if __name__ == '__main__':
    unittest.main()