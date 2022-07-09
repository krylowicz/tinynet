from __future__ import annotations

from functools import partialmethod
from typing import Any

from tinynet.tensor import Tensor, CL_CTX, CL_QUEUE


class Context:
    def __init__(self, op_fn: Function, *tensors: Tensor) -> None:
        self.op_fn = op_fn
        self.cl_ctx = CL_CTX
        self.cl_queue = CL_QUEUE
        self.parents = [tensor for tensor in tensors if type(tensor) is Tensor]
        self.saved_tensors: list[Tensor] = []

    def save_for_backward(self, *tensors: Tensor) -> None:
        for tensor in tensors:
            if type(tensor) is not Tensor:
                raise TypeError(
                    f"Got {type(tensor)} instead of Tensor. Only Tensors should be saved in "
                    f"save_for_backward(). To save constants, save directly as a new attribute."
                )

            self.saved_tensors.append(tensor)


class Function:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        cls = self.__class__
        raise RuntimeError(
            f"{cls} should not be instantiated. Use {cls.__name__}.apply instead."
            f"Ops functions should be defined using static methods."
        )

    @staticmethod
    def forward(ctx: Context, *args: Any, **kwargs: Any) -> Any:
        """
        Performs the forward pass of the function.

        Needs to be overridden by all subclasses.
        """
        raise NotImplementedError("You must implement the forward method for custom ops function")

    @staticmethod
    def backward(ctx: Context, *args: Any, **kwargs: Any) -> Any:
        """
        Defines a formula for differentiating the function on the backward pass.

        Needs to be overridden by all subclasses.
        """
        raise NotImplementedError("You must implement the backward method for custom ops function")

    def apply(self: Tensor, op_fn: Function, *tensors: Tensor, **kwargs: Any) -> Tensor:
        ctx = Context(op_fn, self, *tensors)
        ret = op_fn.forward(ctx, self, *tensors, **kwargs)
        ret._ctx = ctx

        return ret

    # TODO: register already defined, i and r ops
    @staticmethod
    def register(cls: Function = None, gpu: bool = False):
        def _register(cls):
            op_name = cls.__name__.lower()

            if gpu:
                Tensor.ops_gpu[op_name] = cls
            else:
                Tensor.ops_cpu[op_name] = cls

            if op_name in {"sum", "max"}:
                setattr(Tensor, f"_{op_name}", partialmethod(cls.apply, cls))
            else:
                setattr(Tensor, op_name, partialmethod(cls.apply, cls))

            return cls

        if cls:
            return _register(cls)

        return _register
