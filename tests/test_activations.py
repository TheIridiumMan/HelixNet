import pytest
import numpy as np
import src.activations as activs

def ReLU_tests():
    assert np.array_equal(activs.ReLU([-1, 0, 2, -10,5, 15,9]),
                          [0,0,2, 0, 5, 15, 9])
