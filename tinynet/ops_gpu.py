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


@Function.register(gpu=True)
class Add(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        ret = new_cl_buffer(ctx.cl_ctx, determine_shape(x.shape, y.shape))
        prg = cl.Program(ctx.cl_ctx, """
            __kernel void add(__global const float *x, __global const float *y, __global float *ret) {
                int g_id = get_global_id(0);
                ret[g_id] = x[g_id] + y[g_id];
            }
        """).build()
        prg.add(ctx.cl_queue, (ret.size//4,), None, x.data, y.data, ret)
        requires_grad = x.requires_grad or y.requires_grad

        return Tensor(ret, requires_grad=requires_grad)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        return grad_output, grad_output


@Function.register(gpu=True)
class Sub(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        ret = new_cl_buffer(ctx.cl_ctx, determine_shape(x.shape, y.shape))
        prg = cl.Program(ctx.cl_ctx, """
            __kernel void sub(__global const float *x, __global const float *y, __global float *ret) {
                int g_id = get_global_id(0);
                ret[g_id] = x[g_id] - y[g_id];
            }
        """).build()
        prg.sub(ctx.cl_queue, (ret.size//4,), None, x.data, y.data, ret)
        requires_grad = x.requires_grad or y.requires_grad

        return Tensor(ret, requires_grad=requires_grad)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        return grad_output, -grad_output
