import unittest
from tinynet.tensor import Tensor


class TestTensor(unittest.TestCase):
    def test_transpose(self):
        x = Tensor.randn((3, 7))

        self.assertEqual(x.T.shape, (7, 3))
