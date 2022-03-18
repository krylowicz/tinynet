from __future__ import annotations
from functools import partialmethod
from typing import Any, List
from tinynet.tensor import Tensor


class Context:
    def __init__(self, op_fn: Function, *tensors: List[Tensor]) -> None:
        self.op_fn = op_fn
        self.parents = tensors
        self.saved_tensors = []

    def save_for_backward(self, *tensors: List[Tensor]) -> None:
        self.saved_tensors.extend(*tensors)


class Function:
    def __init__(self, *args, **kwargs) -> None:
        cls = self.__class__
        raise RuntimeError(
            f"{cls} should not be instantiated. Use {cls.__name__}.apply instead."
            f"Ops functions should be defined using static methods."
        )

    @staticmethod
    def forward(ctx: Context, *args: Any, **kwargs: Any) -> Any:
        r"""
        Performs the forward pass of the function.

        Needs to be overridden by all subclasses.
        """
        raise NotImplementedError("You must implement the forward method for custom ops function")

    @staticmethod
    def backward(ctx: Context, *args: Any, **kwargs: Any) -> Any:
        r"""
        Defines a formula for differentiating the function on the backward pass.

        Needs to be overridden by all subclasses.
        """
        raise NotImplementedError("You must implement the backward method for custom ops function")

    def apply(self, op_fn: Function, *tensors: List[Tensor]) -> Tensor:
        ctx = Context(op_fn, self, *tensors)
        ret = Tensor(op_fn.forward(ctx, self.data, *[t.data for t in tensors]))
        ret._ctx = ctx

        return ret


def register(name: str, op_fn: Function) -> None:
    setattr(Tensor, name, partialmethod(op_fn.apply, op_fn))