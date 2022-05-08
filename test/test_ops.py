import torch
import numpy as np
from tinynet.tensor import Tensor
import unittest


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


class TestOps(unittest.TestCase):
    def test_add(self):
        helper([(25, 25), (25, 25)], Tensor.add, torch.add)

    def test_sub(self):
        helper([(25, 25), (25, 25)], Tensor.sub, torch.sub)

    def test_mul(self):
        helper([(25, 25), (25, 25)], Tensor.mul, torch.mul)

    def test_matmul(self):
        helper([(25, 25), (25, 25)], Tensor.dot, torch.matmul)

    def test_relu(self):
        helper([(10, 10)], Tensor.relu, torch.relu)

    def test_pow(self):
        helper([(10, 10), (10, 10)], Tensor.pow, torch.pow)

    def test_softmax(self):
        helper([(1, 10)], Tensor.softmax, lambda x: torch.softmax(x, dim=1))
        helper([(32, 10)], Tensor.softmax, lambda x: torch.softmax(x, dim=1))
