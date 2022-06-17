import numpy as np
import unittest

import torch
import torch.optim as torch_optim

import tinynet.optim as optim
from tinynet.tensor import Tensor


x_init = np.random.randn(1, 3)
y_init = np.random.randn(3, 3)
z_init = np.random.randn(1, 3)


class TinyNet:
    def __init__(self):
        self.x = Tensor(x_init, requires_grad=True)
        self.y = Tensor(y_init, requires_grad=True)
        self.z = Tensor(z_init)

    def forward(self):
        out = self.x.dot(self.y).relu()
        out = out.logsoftmax()
        out = out.mul(self.z).add(self.z).sum()

        return out


class TorchNet:
    def __init__(self):
        self.x = torch.tensor(x_init, requires_grad=True)
        self.y = torch.tensor(y_init, requires_grad=True)
        self.z = torch.tensor(z_init)

    def forward(self):
        out = self.x.matmul(self.y).relu()
        out = out.log_softmax(dim=1)
        out = out.mul(self.z).add(self.z).sum()

        return out


def tiny_optimizer(optimizer, **kwargs):
    model = TinyNet()
    optimizer = optimizer([model.x, model.y], **kwargs)
    out = model.forward()
    out.backward()
    optimizer.step()

    return model.x.data, model.y.data


def torch_optimizer(optimizer, **kwargs):
    model = TorchNet()
    optimizer = optimizer([model.x, model.y], **kwargs)
    out = model.forward()
    out.backward()
    optimizer.step()

    return model.x.detach().numpy(), model.y.detach().numpy()


class TestOptim(unittest.TestCase):
    def test_sgd(self):
        for x, y in zip(tiny_optimizer(optim.SGD, lr=0.01), torch_optimizer(torch_optim.SGD, lr=0.01)):
            np.testing.assert_allclose(x, y)

    def test_optim(self):
        for x, y in zip(
            tiny_optimizer(optim.RMSProp, lr=0.001, decay=0.9, eps=1e-8),
            torch_optimizer(torch_optim.RMSprop, lr=0.001, alpha=0.9, eps=1e-8)
        ):
            np.testing.assert_allclose(x, y)

    def test_adam(self):
        for x, y in zip(tiny_optimizer(optim.Adam), torch_optimizer(torch_optim.Adam)):
            np.testing.assert_allclose(x, y)
