import numpy as np

from tinynet.tensor import Tensor
from tinynet.nn.module import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        weight = np.random.uniform(-1., 1., size=(in_features, out_features)) / np.sqrt(out_features * in_features)
        bias = np.random.rand(out_features) - 0.5

        self.weight = Tensor(weight, requires_grad=True, is_parameter=True)
        self.bias = Tensor(bias, requires_grad=True, is_parameter=True)

    def forward(self, x: Tensor) -> Tensor:
        return x.dot(self.weight) + self.bias
