import numpy as np
import pyopencl as cl

from tinynet.tensor import Tensor
from tinynet.function import Function, Context


def new_cl_buffer(ctx: cl.Context, output_shape: tuple[int, ...]) -> cl._cl.Buffer:
    buffer = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, np.prod(output_shape) * 4)
    buffer.shape = output_shape

    return buffer


def determine_shape(x_shape: tuple[int, ...], y_shape: tuple[int, ...]) -> tuple[int, ...]:
    return np.broadcast_shapes(x_shape, y_shape)


def binary_op(ctx: Context, name: str, op_fn: str, x: Tensor, y: Tensor) -> Tensor:
    ret = new_cl_buffer(ctx.cl_ctx, determine_shape(x.shape, y.shape))
    prg = cl.Program(ctx.cl_ctx, f"""
        __kernel void {name}(__global const float *x, __global const float *y, __global float *ret) {{
            int g_id = get_global_id(0);
            ret[g_id] = x[gid] {op_fn} y[gid]; 
        }}
    """).build()
    prg.__dict__[name](ctx.cl_queue, (ret.size // 4,), None, x.data, y.data, ret)
    requires_grad = x.requires_grad or y.requires_grad

    return Tensor(ret, requires_grad=requires_grad)


@Function.register(gpu=True)
class Add(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        return binary_op(ctx, "add", "+", x, y)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        return grad_output, grad_output


@Function.register(gpu=True)
class Sub(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        return binary_op(ctx, "sub", "-", x, y)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        return grad_output, -grad_output


@Function.register(gpu=True)
class Mul(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        return binary_op(ctx, "mul", "*", x, y)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        return grad_output, grad_output
