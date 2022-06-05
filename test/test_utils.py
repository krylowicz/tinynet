import unittest
import numpy as np
from tinynet.utils import to_one_hot
from tinynet.tensor import Tensor


class TestUtils(unittest.TestCase):
    def test_to_one_hot(self):
        tiny_one_hot = to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
        corr_one_hot = [[0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 0, 0]]

        self.assertTrue(np.array_equal(tiny_one_hot.data, np.array(corr_one_hot)))
