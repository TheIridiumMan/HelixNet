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
        case = layers.Layer("D", [mg.zeros((7, 50))])
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


class DenseTests(unittest.TestCase):
    def test_matrix_init(self):
        mat = layers.Dense(64, 32, lambda x: x, use_bias=False)
        self.assertEqual(mat.trainable_params[0].data.shape, (64, 32))
        mat = layers.Dense(64, 32, lambda x: x, use_bias=True)
        self.assertEqual(mat.trainable_params[0].data.shape, (64, 32))
        self.assertEqual(mat.trainable_params[1].data.shape, (1, 32))

    def test_tensor_output_init(self):
        mat = layers.Dense(64, (2, 3, 4), lambda x: x, use_bias=False)
        self.assertEqual(mat.trainable_params[0].data.shape, (64, 2, 3, 4))
        mat = layers.Dense(64, (4, 7, 15), lambda x: x, use_bias=True)
        self.assertEqual(mat.trainable_params[0].data.shape, (64, 4, 7, 15))
        self.assertEqual(mat.trainable_params[1].data.shape, (1, 4, 7, 15))

    def test_tensor_input_init(self):
        mat = layers.Dense((2, 3, 4), 64, lambda x: x, use_bias=False)
        self.assertEqual(mat.trainable_params[0].data.shape, (2, 3, 4, 64))
        mat = layers.Dense((2, 3, 4), 64, lambda x: x, use_bias=True)
        self.assertEqual(mat.trainable_params[0].data.shape, (2, 3, 4, 64))
        self.assertEqual(mat.trainable_params[1].data.shape, (64,))

    def test_tensor_in_out_init(self):
        mat = layers.Dense((2, 3, 4), (5, 3, 4), lambda x: x, use_bias=False)
        self.assertEqual(mat.trainable_params[0].data.shape, (2, 3, 4, 5, 3, 4))
        mat = layers.Dense((2, 3, 4), (5, 3, 4), lambda x: x, use_bias=True)
        self.assertEqual(mat.trainable_params[0].data.shape, (2, 3, 4, 5, 3, 4))
        self.assertEqual(mat.trainable_params[1].data.shape, (5, 3, 4))

    def test_matrix_forward(self):
        mat = layers.Dense(64, 32, lambda x: x, use_bias=False)
        result = mat.forward(np.ones((4, 64)))
        self.assertEqual(result.shape, (4, 32))
        loss_val = loss.MeanAbsError(result, np.zeros((4, 32)))
        loss_val.backward()
        for param in mat.trainable_params:
            self.assertIsNotNone(param.grad)

    def test_tensor_forward(self):
        mat = layers.Dense((64, 86), (32, 45), lambda x: x, use_bias=False)
        result = mat.forward(np.ones((4, 64, 86)))
        self.assertEqual(result.shape, (4, 64, 86))
        loss_val = loss.MeanAbsError(result, np.zeros((4, 32, 45)))
        loss_val.backward()
        for param in mat.trainable_params:
            self.assertIsNone(param.grad)

    def test_output_shape(self):
        mat = layers.Dense(64, 32, lambda x: x, use_bias=False)
        self.assertEqual(mat.output_shape(), (32,))
        mat = layers.Dense(64, 32, lambda x: x, use_bias=True)
        self.assertEqual(mat.output_shape(), (32,))

        mat = layers.Dense((2, 3, 4), 64, lambda x: x, use_bias=False)
        self.assertEqual(mat.output_shape(), (64,))
        mat = layers.Dense((2, 3, 4), 64, lambda x: x, use_bias=True)
        self.assertEqual(mat.output_shape(), (64,))

        mat = layers.Dense(64, (2, 3, 4), lambda x: x, use_bias=False)
        self.assertEqual(mat.output_shape(), (2,3,4))
        mat = layers.Dense( 64,(2, 3, 4), lambda x: x, use_bias=True)
        self.assertEqual(mat.output_shape(), (2,3,4))


if __name__ == '__main__':
    unittest.main()
