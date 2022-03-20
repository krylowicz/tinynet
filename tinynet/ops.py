import numpy as np
from typing import Tuple
from tinynet.function import Function, Context
from tinynet.tensor import Tensor


@Function.register
class Add(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        return x + y

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, grad_output


@Function.register
class Sub(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        return x - y

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, -grad_output


@Function.register
class Mul(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        ctx.save_for_backward(x, y)

        return x * y

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = ctx.saved_tensors

        return y * grad_output, x * grad_output


@Function.register
class Pow(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        ctx.save_for_backward(x, y)

        return x ** y

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = ctx.saved_tensors

        return y * (x ** (y - 1)) * grad_output, np.log(x) * (x ** y) * grad_output


@Function.register
class Sum(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)

        return np.array([x.sum()])

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        x, = ctx.saved_tensors

        return grad_output * np.ones_like(x)


@Function.register
class Dot(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        ctx.save_for_backward(x, y)
        return x.dot(y)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = ctx.saved_tensors

        return grad_output.dot(y.T), x.T.dot(grad_output)