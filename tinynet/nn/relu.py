from tinynet.tensor import Tensor
from tinynet.nn.module import Module


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
