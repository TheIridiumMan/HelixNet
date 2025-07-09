import unittest
from unittest.mock import Mock

import numpy as np

from helixnet import optimizers


class NesterovSGD(unittest.TestCase):
    def test_primary(self):
        optim = optimizers.NesterovSGD(0.1)
        param = Mock()
        param.data = np.ones((10, 10))
        param.grad = np.ones((10, 10))

        test_result = np.empty_like(param.data)
        test_result.fill(0.81)

        optim.optimize_param(param)
        self.assertTrue((param.data == test_result).all())

    def test_two_runs(self):
        param = Mock()
        param.data = np.ones((10, 10))
        param.grad = np.ones((10, 10))

        optim = optimizers.NesterovSGD(0.1)
        optim.optimize_param(param)

        optim.optimize_param(param)

        test_result = np.empty_like(param.data)
        test_result.fill(0.5390000000000001)

        self.assertTrue((param.data == test_result).all())

