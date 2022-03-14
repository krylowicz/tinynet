import numpy as np
from typing import Tuple
from tinynet.function import Function
from tinynet.tensor import Tensor


class Add(Function):
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor) -> Tensor:
        return x + y

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, grad_output


class Sub(Function):
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor) -> Tensor:
        return x - y

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, -grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor) -> Tensor:
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = ctx.saved_tensors

        return y * grad_output, x * grad_output


class Pow(Function):
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor) -> Tensor:
        ctx.save_for_backward(x, y)
        return x ** y

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = ctx.saved_tensors

        return y * (x ** (y - 1)) * grad_output, np.log(x) * (x ** y) * grad_output
