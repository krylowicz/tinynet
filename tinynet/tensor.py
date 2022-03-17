from __future__ import annotations

from typing import Optional
import numpy as np


class Tensor:
    def __init__(self, data: np.ndarray | list, requires_grad: bool = False) -> None:
        self._data = data if isinstance(data, np.ndarray) else np.array(data)
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None

        if self.requires_grad:
            self.zero_grad()

        # context for backpropagation
        self._ctx = None

    def __repr__(self) -> str:
        return f"<Tensor with shape {self.shape}, requires_grad={self.requires_grad}>"

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def data(self) -> np.ndarray:
        return self._data

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

        assert self.requires_grad, "Attempted to call backward on a non-requires_grad Tensor"

