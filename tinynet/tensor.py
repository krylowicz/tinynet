from __future__ import annotations

import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tinynet.function import Context


class Tensor:
    def __init__(self, data: np.ndarray | list[int, float], requires_grad: bool = False) -> None:
        self._data = data.astype(np.float32) if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self._grad: Optional[Tensor] = None

        if self.requires_grad:
            self.zero_grad()

        # context for backpropagation
        self._ctx: Optional[Context] = None

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def grad(self) -> Optional[Tensor]:
        return self._grad

    @property
    def data(self) -> np.ndarray:
        return self._data

    @grad.setter
    def grad(self, value: Tensor | np.ndarray) -> None:
        self._grad = value if isinstance(value, Tensor) else Tensor(value)

    @data.setter
    def data(self, data: np.ndarray | list) -> None:
        self._data = data if isinstance(data, np.ndarray) else np.array(data)

        if self.requires_grad:
            self.zero_grad()

    def __repr__(self) -> str:
        return f"""{self.data}, requires_grad={self.requires_grad}
        {f", grad_fn={self._ctx.op_fn}" if self._ctx is not None else ''}"""

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, key: slice | tuple) -> Tensor:
        return Tensor(self.data[key])

    def __setitem__(self, key: slice | tuple, value: Tensor | np.ndarray) -> None:
        self.data[key] = value.data if isinstance(value, Tensor) else value

    def zero_grad(self) -> None:
        self.grad = np.ones_like(self.data)

    # -- binary ops --

    def add(self, other: Tensor) -> Tensor:
        return Tensor.add(self, other)

    def __add__(self, other: Tensor) -> Tensor:
        return Tensor.add(self, other)

    __radd__ = __add__

    def sub(self, other: Tensor) -> Tensor:
        return Tensor.sub(self, other)

    def __sub__(self, other: Tensor) -> Tensor:
        return Tensor.sub(self, other)

    __rsub__ = __sub__

    def mul(self, other: Tensor) -> Tensor:
        return Tensor.mul(self, other)

    def __mul__(self, other: Tensor) -> Tensor:
        return Tensor.mul(self, other)

    __rmul__ = __mul__

    def __neg__(self) -> Tensor:
        self.data = -self.data
        return self

    def __matmul__(self, other: Tensor) -> Tensor:
        return Tensor.dot(self, other)

    def pow(self, power: Tensor) -> Tensor:
        return Tensor.pow(self, power)

    def __pow__(self, power: Tensor) -> Tensor:
        return Tensor.pow(self, power)

    def dot(self, other: Tensor) -> Tensor:
        return self.__matmul__(other)

    def mean(self) -> Tensor:
        return self.sum().mul(Tensor(np.array([1 / np.prod(self.shape)])))

    # -- reduce ops --

    def sum(self, axis: int = None) -> Tensor:
        # TODO: add support for axis in op class
        return Tensor.sum(self, axis)

    # -- unary ops --

    def relu(self) -> Tensor:
        return Tensor.relu(self)

    def softmax(self) -> Tensor:
        return Tensor.softmax(self)

    def logsoftmax(self) -> Tensor:
        return Tensor.logsoftmax(self)

    # -- autograd engine --

    def backward(self) -> None:
        if self._ctx is None:
            return

        if self.requires_grad is False:
            raise RuntimeError("Attempted to call backward on a non-requires_grad Tensor")

        current_node_grad = self._ctx.op_fn.backward(self._ctx, self.grad)
        for i, parent in enumerate(parents := self._ctx.parents):
            if len(parents) == 1:
                current_node_grad = np.expand_dims(current_node_grad.data, axis=0)
            parent.grad = current_node_grad[i]
            parent.backward()

    # -- creation helpers --

    @classmethod
    def zeros(cls, shape: tuple) -> Tensor:
        return cls(np.zeros(shape))

    @classmethod
    def ones(cls, shape: tuple) -> Tensor:
        return cls(np.ones(shape))

    @classmethod
    def randn(cls, shape: tuple) -> Tensor:
        return cls(np.random.randn(*shape))
