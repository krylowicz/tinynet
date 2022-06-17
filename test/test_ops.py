import torch
import unittest
import numpy as np

from tinynet.tensor import Tensor


def helper(shapes, tiny_op, torch_op):
    torch.manual_seed(2137)

    torch_tensors = [torch.tensor(np.random.random(size=shape), requires_grad=True) for shape in shapes]
    tiny_tensors = [Tensor(data.detach().numpy(), requires_grad=True) for data in torch_tensors]

    tiny_res = tiny_op(*tiny_tensors)
    torch_res = torch_op(*torch_tensors)

    np.testing.assert_allclose(tiny_res.data, torch_res.detach().numpy(), atol=1e-5)

    print("forward pass passed")

    tiny_res.sum().backward()
    torch_res.sum().backward()

    for tiny_grad, torch_grad in zip(tiny_tensors, torch_tensors):
        print(tiny_grad.grad.shape, torch_grad.grad.shape)
        np.testing.assert_allclose(tiny_grad.grad.data, torch_grad.grad.detach().numpy(), atol=1e-5)


class TestUnaryOps(unittest.TestCase):
    def test_relu(self):
        helper([(10, 10)], Tensor.relu, torch.relu)

    def test_exp(self):
        helper([(10, 10)], Tensor.exp, torch.exp)

    def test_log(self):
        helper([(10, 10)], Tensor.log, torch.log)

    def test_softmax(self):
        helper([(1, 10)], Tensor.softmax, lambda x: torch.softmax(x, dim=1))
        helper([(32, 10)], Tensor.softmax, lambda x: torch.softmax(x, dim=1))

    def test_logsoftmax(self):
        helper([(1, 10)], Tensor.logsoftmax, lambda x: torch.log_softmax(x, dim=1))
        helper([(32, 10)], Tensor.logsoftmax, lambda x: torch.log_softmax(x, dim=1))


class TestBinaryOps(unittest.TestCase):
    def test_add(self):
        helper([(25, 25), (25, 25)], Tensor.add, torch.add)

    def test_sub(self):
        helper([(25, 25), (25, 25)], Tensor.sub, torch.sub)

    def test_mul(self):
        helper([(25, 25), (25, 25)], Tensor.mul, torch.mul)

    def test_matmul(self):
        helper([(25, 25), (25, 25)], Tensor.dot, torch.matmul)

    def test_pow(self):
        helper([(10, 10), (10, 10)], Tensor.pow, torch.pow)


class TestReduceOps(unittest.TestCase):
    def test_sum(self):
        helper([(32, 16, 16, 3)], Tensor.sum, torch.sum)
        helper([(32, 16, 16, 3)], lambda x: x.sum(axis=0), lambda x: torch.sum(x, dim=0))
        # helper([(32, 16, 16, 3)], lambda x: x.sum(axis=3), lambda x: torch.sum(x, dim=3))
        # helper([(32, 16, 16, 3)], lambda x: x.sum(axis=(1, 3)), lambda x: torch.sum(x, dim=(1, 3)))

    def test_max(self):
        pass


class TestMovementOps(unittest.TestCase):
    def test_reshape(self):
        helper([(2, 3, 4)], lambda x: x.reshape((2, 12)), lambda x: torch.reshape(x, (2, 12)))

    def test_permute(self):
        helper([(6, 5, 4)], lambda x: x.permute((2, 0, 1)), lambda x: torch.permute(x, (2, 0, 1)))

    def test_transpose(self):
        helper([(2, 3)], lambda x: x.transpose((1, 0)), lambda x: torch.permute(x, (1, 0)))

    def test_slice(self):
        pass


class TestProcessingOps(unittest.TestCase):
    def test_conv1d(self):
        pass

    def test_conv2d(self):
        pass
