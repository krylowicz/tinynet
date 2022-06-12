from typing import Iterable

from tinynet.tensor import Tensor
from tinynet.nn.module import Module


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = layers

        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def __iter__(self) -> Iterable[Module]:
        yield from self.layers

    def __getitem__(self, idx: int) -> Module:
        return self.layers[idx]

    def train(self) -> None:
        self.is_train = True

        for submodule in self._submodules.values():
            submodule.train()

    def eval(self) -> None:
        self.is_train = False

        for submodule in self._submodules.values():
            submodule.eval()

    def forward(self, x: Tensor) -> Tensor:
        output = x

        for layer in self.layers:
            output = layer.forward(output)

        return output
