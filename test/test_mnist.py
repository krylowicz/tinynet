from __future__ import annotations

import numpy as np
from tqdm import trange
from tinynet.tensor import Tensor
from tinynet.optim import SGD


def init_layer(*shape) -> Tensor:
    return Tensor(np.random.randn(*shape) / np.sqrt(shape[0]), requires_grad=True)


class TinyNet:
    def __init__(self):
        self.l1 = init_layer(784, 128)
        self.l2 = init_layer(128, 10)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    # TODO: add model.parameters
    @property
    def parameters(self) -> Tensor:
        pass

    def forward(self, x: Tensor) -> Tensor:
        x = x.dot(self.l1)
        x.relu(x)
        x = x.dot(self.l2)
        x.logsoftmax(x)

        return x


model = TinyNet()
optim = SGD(model.parameters, lr=0.1)
for i in (t := trange(range(100))):
    # TODO: add mnist loader
    x = Tensor(np.random.randn(1, 784))
    y = model(x)

    loss = -y.log_softmax(y).sum()
    loss.backward()



    # t.set_description(f"epoch {i}, loss {10}, accuracy {10}")