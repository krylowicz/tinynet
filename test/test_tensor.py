import unittest
import torch
import numpy as np

from tinynet.tensor import Tensor


class TestTensor(unittest.TestCase):
    def test_transpose(self):
        x = Tensor.randn(3, 7)

        self.assertEqual(x.T.shape, (7, 3))

    def test_backward(self):
        xi = np.random.randn(1, 3)
        Wi = np.random.randn(3, 3)
        bi = np.random.randn(1, 3)

        def test_tiny():
            x = Tensor(xi, requires_grad=True)
            W = Tensor(Wi, requires_grad=True)
            b = Tensor(bi)

            out = x.dot(W).relu()
            out = out.logsoftmax()
            out = out.mul(b).add(b).sum()
            out.backward()

            return out.data, x.grad.data, W.grad.data

        def test_torch():
            x = torch.tensor(xi, requires_grad=True)
            W = torch.tensor(Wi, requires_grad=True)
            b = torch.tensor(bi)

            out = x.matmul(W).relu()
            out = torch.log_softmax(out, dim=1)
            out = out.mul(b).add(b).sum()
            out.backward()

            return out.detach().numpy(), x.grad.detach().numpy(), W.grad.detach().numpy()

        for tiny_res, torch_res in zip(test_tiny(), test_torch()):
            np.testing.assert_allclose(tiny_res, torch_res, atol=1e-5)
