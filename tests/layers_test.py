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
        loss.MeanAbsError(case.predict(np.ones((1, 64))), np.ones((1, 86)))\
            .backward()
        for param in case.trainable_params:
            self.assertIsNone(param.grad)

    def test_output_shape(self):
        case = layers.Conv2D(1, 16, 3, activation=(lambda x: x))
        # Because Convolution uses the inherited method and doesn't provide it's own
        self.assertEqual(case.output_shape([1, 28, 28]), (16, 26, 26))

    def test_params_amount(self):
        layer1 = layers.Dense(256, 128, lambda x: x)
        self.assertEqual(layer1.total_params(), 256 * 128 + 128)


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
        self.assertEqual(mat.trainable_params[1].data.shape, (1, 64))

    def test_tensor_in_out_init(self):
        mat = layers.Dense((2, 3, 4), (5, 3, 4), lambda x: x, use_bias=False)
        self.assertEqual(mat.trainable_params[0].data.shape, (2, 3, 4, 5, 3, 4))
        mat = layers.Dense((2, 3, 4), (5, 3, 4), lambda x: x, use_bias=True)
        self.assertEqual(mat.trainable_params[0].data.shape, (2, 3, 4, 5, 3, 4))
        self.assertEqual(mat.trainable_params[1].data.shape, (1, 5, 3, 4))

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
        result = mat.forward(np.ones((4, 64, 32)))
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
        self.assertEqual(mat.output_shape(), (2, 3, 4))
        mat = layers.Dense(64, (2, 3, 4), lambda x: x, use_bias=True)
        self.assertEqual(mat.output_shape(), (2, 3, 4))


class MiscLayerTest(unittest.TestCase):
    def test_input_shape_outputShape(self):
        layer = layers.InputShape([1, 28, 28])
        self.assertEqual(layer.output_shape(), (1, 28, 28))

    def test_input_shape_output(self):
        layer = layers.InputShape([1, 28, 28])
        X = np.random.randn(10, 1, 28, 28)
        output = layer.forward(X)
        self.assert_((X == output).all())

    def test_input_shape_strictness(self):
        with self.assertRaises(ValueError):
            layer = layers.InputShape([1, 28, 28])
            X = np.random.randn(10, 50, 16, 28)
            output = layer.forward(X)
        layer = layers.InputShape([1, 28, 28])
        X = np.random.randn(10, 1, 28, 28)
        output = layer.forward(X)

    def test_flatten_creation(self):
        layer = layers.Flatten()
        self.assertEqual(layer.trainable_params, [])
        self.assertEqual(layer.type, layer.name)
        layer = layers.Flatten()
        self.assertEqual(layer.type, layer.name)
        layer = layers.Flatten()
        self.assertEqual(layer.type, layer.name)

    def test_flatten_forward(self):
        layer = layers.Flatten()
        self.assertEqual(layer.forward(np.random.randn(78, 15, 32, 44)).shape,
                         (78, 15 * 32 * 44))


class BatchNormTests(unittest.TestCase):
    def test_params_init(self):
        layer = layers.BatchNorm((784, 128))
        self.assertTupleEqual(layer.weight.shape, (784, 128))
        self.assertTupleEqual(layer.bias.shape, (128,))

        layer = layers.BatchNorm((85, 87))
        self.assertTupleEqual(layer.weight.shape, (85, 87))
        self.assertTupleEqual(layer.bias.shape, (87,))

    def test_forward(self):
        x = np.random.randint(0, 500, size=(784, 128))
        layer = layers.BatchNorm((784, 128))
        y = layer.forward(x)
        self.assertFalse((x == y).all())


class DropoutTests(unittest.TestCase):
    def test_forward(self):
        x = np.random.randint(1, 500, size=(784, 128))
        layer = layers.Dropout(0.2)
        y = layer.forward(x)
        # This assert ensures that x and y aren't identically equally
        self.assertTrue((x != y).any())
        self.assertTrue(np.allclose(np.count_nonzero(y) /
                                    (784 * 128), 0.8, 1e-2))

    def test_predict(self):
        x = np.random.randint(1, 500, size=(784, 128))
        layer = layers.Dropout(0.2)
        y = layer.predict(x)
        self.assertTrue((x == y).all())


if __name__ == '__main__':
    unittest.main()
