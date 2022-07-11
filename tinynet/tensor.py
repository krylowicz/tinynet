from __future__ import annotations

import os

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinynet.function import Context, Function

GPU = int(os.getenv("GPU", 0))

CL_CTX = None
CL_QUEUE = None

if GPU:
    import pyopencl as cl
    devices = cl.get_platforms()[0].get_devices(cl.device_type.GPU)

    if len(devices) == 0:
        raise RuntimeError("No GPU found")

    CL_CTX = cl.Context(devices=devices)
    CL_QUEUE = cl.CommandQueue(CL_CTX)


class Tensor:
    ops_cpu: dict[str, Function] = {}
    ops_gpu: dict[str, Function] = {}

    def __init__(
        self,
        data: np.ndarray | list[int | float],
        requires_grad: bool = False,
        is_parameter: bool = False
    ) -> None:
        self._gpu = False
        self._data = self._assign_data(data)
        self._grad: Tensor | None = None
        self.requires_grad = requires_grad
        self.is_parameter = is_parameter

        if GPU:
            self.gpu()

        if self.requires_grad:
            self.zero_grad()

        # context for backpropagation
        self._ctx: Context | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def grad(self) -> Tensor | None:
        return self._grad

    @grad.setter
    def grad(self, value: Tensor | np.ndarray) -> None:
        self._grad = value if isinstance(value, Tensor) else Tensor(value)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, data: np.ndarray | list | cl._cl.Buffer) -> None:
        self._data = self._assign_data(data)

    def _assign_data(self, data):
        if isinstance(data, list):
            self._gpu = False

            return np.array(data).astype(np.float32)
        elif isinstance(data, np.ndarray):
            self._gpu = False

            return data.astype(np.float32)
        elif GPU and isinstance(data, cl._cl.Buffer):
            self._gpu = True

            return data
        else:
            try:
                data = np.array(data).astype(np.float32)
                self._gpu = False

                return data
            except Exception:
                raise TypeError(f"Tensor data must be list, numpy.ndarray or cl.Buffer. got {type(data)}")

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def __repr__(self) -> str:
        return f"""{self.data}, requires_grad={self.requires_grad}
        {f", grad_fn={self._ctx.op_fn}" if self._ctx is not None else ''}"""

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, key: slice | tuple) -> Tensor:
        return Tensor(self.data[key], requires_grad=self.requires_grad, is_parameter=self.is_parameter)

    def __setitem__(self, key: slice | tuple, value: Tensor | np.ndarray) -> None:
        self.data[key] = value.data if isinstance(value, Tensor) else value

    def zero_grad(self) -> None:
        self.grad = np.ones_like(self.data)

    # -- gpu --

    def gpu(self) -> Tensor:
        if not GPU:
            raise RuntimeWarning("GPU is not available. set GPU=1 to enable")

        if not self._gpu:
            data = cl.Buffer(CL_CTX, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data)
            data.shape = self.shape

            return Tensor(data, requires_grad=self.requires_grad, is_parameter=self.is_parameter)

        return self

    def cpu(self) -> Tensor:
        if self._gpu:
            data = np.empty(self.shape, dtype=np.float32)
            cl.enqueue_copy(CL_QUEUE, data, self.data)
            self.data = data
            self._gpu = False

        return self

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

    def div(self, other: Tensor) -> Tensor:
        return self * (other ** Tensor(np.array(-1.0)))

    __truediv__ = div

    def __neg__(self) -> Tensor:
        self.data = -self.data

        return self

    def matmul(self, other: Tensor) -> Tensor:
        return Tensor.matmul(self, other)

    def __matmul__(self, other: Tensor) -> Tensor:
        return Tensor.matmul(self, other)

    def dot(self, other: Tensor) -> Tensor:
        return self.__matmul__(other)

    def pow(self, power: Tensor) -> Tensor:
        return Tensor.pow(self, power)

    def __pow__(self, power: Tensor) -> Tensor:
        return Tensor.pow(self, power)

    def exp(self) -> Tensor:
        return Tensor.exp(self)

    def log(self) -> Tensor:
        return Tensor.log(self)

    def transpose(self, axis: tuple[int, int] = (1, 0)) -> Tensor:
        return self.permute(axis)

    # -- reduce ops --

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        return self._sum(axis=axis, keepdims=keepdims)

    def max(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        return self._max(axis=axis, keepdims=keepdims)

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out = self.sum(axis=axis, keepdims=keepdims)
        return out * Tensor((np.prod(out.shape) / np.prod(self.shape)))

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
    def zeros(cls, *shape: int, **kwargs) -> Tensor:
        return cls(np.zeros(*shape), **kwargs)

    @classmethod
    def ones(cls, *shape: int, **kwargs) -> Tensor:
        return cls(np.ones(*shape), **kwargs)

    @classmethod
    def randn(cls, *shape: int, **kwargs) -> Tensor:
        return cls(np.random.randn(*shape), **kwargs)
