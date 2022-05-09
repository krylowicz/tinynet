from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from tinynet.tensor import Tensor


class Optimizer:
    def __init__(self, params: Iterable[Tensor]) -> None:
        self.params = list(params)

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad.zero_grad()
