import numpy as np
from typing import Tuple
from tinynet.tensor import Tensor
from tinynet.function import Function, Context


# unary ops
@Function.register
class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)

        return Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        x, = ctx.saved_tensors

        grad = grad_output.data * (x.data >= 0)

        return Tensor(grad)


@Function.register
class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor) -> Tensor:
        e_x = np.exp(x.data - np.max(x.data, axis=1, keepdims=True))
        softmax = e_x / np.sum(e_x, axis=1, keepdims=True)

        ctx.save_for_backward(softmax)

        return Tensor(softmax, requires_grad=x.requires_grad)

    @staticmethod
    def backward(ctx: Context, g: Tensor) -> Tensor:
        softmax, = ctx.saved_tensors

        def softmax_vector_derivative(softmax: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
            shape = softmax.shape

            softmax = np.reshape(softmax, (1, -1))
            grad_output = np.reshape(grad_output, (1, -1))

            d_softmax = softmax * np.identity(softmax.size) - softmax.T @ softmax

            return (grad_output @ d_softmax).reshape(shape)

        grad = np.empty_like(softmax)
        for i in range(softmax.shape[0]):
            grad[i] = softmax_vector_derivative(softmax[i], g.data[i])

        return Tensor(grad)


# binary ops
@Function.register
class Add(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        ctx.save_for_backward(x, y)

        requires_grad = x.requires_grad or y.requires_grad

        return Tensor(x.data + y.data, requires_grad=requires_grad)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = ctx.saved_tensors

        grad_x = Tensor(np.ones(x.shape) * grad_output.data)
        grad_y = Tensor(np.ones(y.shape) * grad_output.data)

        return grad_x, grad_y


@Function.register
class Sub(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        ctx.save_for_backward(x, y)

        requires_grad = x.requires_grad or y.requires_grad

        return Tensor(x.data - y.data, requires_grad=requires_grad)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = ctx.saved_tensors

        grad_x = Tensor(np.ones(x.shape) * grad_output.data)
        grad_y = Tensor(-np.ones(y.shape) * grad_output.data)

        return grad_x, grad_y


@Function.register
class Pow(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        ctx.save_for_backward(x, y)

        requires_grad = x.requires_grad or y.requires_grad

        return Tensor(x.data ** y.data, requires_grad=requires_grad)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = ctx.saved_tensors

        x = x.data
        y = y.data

        grad_x = Tensor(y * (x ** (y - 1)) * grad_output.data)
        grad_y = Tensor(x ** y * np.log(x) * grad_output.data)

        return grad_x, grad_y


@Function.register
class Mul(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        ctx.save_for_backward(x, y)

        requires_grad = x.requires_grad or y.requires_grad

        return Tensor(x.data * y.data, requires_grad=requires_grad)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = ctx.saved_tensors

        grad_x = Tensor(y.data * grad_output.data)
        grad_y = Tensor(x.data * grad_output.data)

        return grad_x, grad_y


@Function.register
class Dot(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        ctx.save_for_backward(x, y)

        requires_grad = x.requires_grad or y.requires_grad

        return Tensor(x.data @ y.data, requires_grad=requires_grad)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = ctx.saved_tensors

        grad_x = Tensor(grad_output.data @ y.data.T)
        grad_y = Tensor(x.data.T @ grad_output.data)

        return grad_x, grad_y


# reduce ops
@Function.register
class Sum(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)

        return Tensor(x.data.sum(keepdims=True), requires_grad=x.requires_grad)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        x, = ctx.saved_tensors

        grad = np.broadcast_to(grad_output.data, x.shape)

        return Tensor(grad)
