from __future__ import annotations
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
        self._ctx: Context | None = None

    def __repr__(self) -> str:
        return f'{self.data}{f", requires_grad={self.requires_grad}"}{f", grad_fn={self._ctx.op_fn}" if self._ctx is not None else ""}'

    def __str__(self) -> str:
        return self.__repr__()

    def __add__(self, other: Tensor) -> Tensor:
        return Tensor.add(self, other)

    __radd__ = __add__

    def __sub__(self, other: Tensor) -> Tensor:
        return Tensor.sub(self, other)

    __rsub__ = __sub__

    def __mul__(self, other: Tensor) -> Tensor:
        return Tensor.mul(self, other)

    __rmul__ = __mul__

    def __matmul__(self, other: Tensor) -> Tensor:
        return Tensor.dot(self, other)

    def __pow__(self, power: Tensor) -> Tensor:
        return Tensor.pow(self, power)

    def __getitem__(self, key: slice | tuple) -> Tensor:
        return Tensor(self.data[key])

    def __setitem__(self, key: slice | tuple, value: Tensor | np.ndarray) -> None:
        self.data[key] = value.data if isinstance(value, Tensor) else value

    def dot(self, other: Tensor) -> Tensor:
        return self.__matmul__(other)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def grad(self) -> np.ndarray:
        return self._grad.data

    @property
    def data(self) -> np.ndarray:
        return self._data

    @grad.setter
    def grad(self, value: np.ndarray) -> None:
        self._grad = Tensor(value)

    @data.setter
    def data(self, data: np.ndarray | list) -> None:
        self._data = data if isinstance(data, np.ndarray) else np.array(data)

        if self.requires_grad:
            self.zero_grad()

    @classmethod
    def zeros(cls, shape: tuple) -> Tensor:
        return cls(np.zeros(shape))

    @classmethod
    def ones(cls, shape: tuple) -> Tensor:
        return cls(np.ones(shape))

    @classmethod
    def randn(cls, shape: tuple) -> Tensor:
        return cls(np.random.randn(*shape))

    def zero_grad(self) -> None:
        self.grad = np.ones_like(self.data)

    def backward(self) -> None:
        if self._ctx is None:
            return

        if self.requires_grad is False:
            raise ValueError("Attempted to call backward on a non-requires_grad Tensor")

        current_node_grad = self._ctx.op_fn.backward(self._ctx, self.grad)
        for i, parent in enumerate(parents := self._ctx.parents):
            if len(parents) == 1:
                current_node_grad = np.expand_dims(current_node_grad, axis=1)
            parent.grad = current_node_grad[i]
            parent.backward()


# TODO: register op functions in a better way
from tinynet.ops import *
