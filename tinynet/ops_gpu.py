import numpy as np
import pyopencl as cl

from tinynet.tensor import Tensor
from tinynet.function import Function, Context


@Function.register(gpu=True)
class Add(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        ret = cl.Buffer(ctx.cl_ctx, cl.mem_flags.WRITE_ONLY, x.data.size * 4)
        ret.shape = x.shape
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
        ret = cl.Buffer(ctx.cl_ctx, cl.mem_flags.WRITE_ONLY, x.data.size * 4)
        ret.shape = x.shape
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
