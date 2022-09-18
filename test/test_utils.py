import pytest
import unittest
import numpy as np

from tinynet.constants import GPU
from tinynet.tensor import Tensor
from tinynet.utils import to_one_hot


class TestUtils(unittest.TestCase):
    @pytest.mark.skipif(GPU, reason="this op is not implemented on GPU")
    def test_to_one_hot(self):
        tiny_one_hot = to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
        corr_one_hot = [[0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 0, 0]]

        self.assertTrue(np.array_equal(tiny_one_hot.data, np.array(corr_one_hot)))
