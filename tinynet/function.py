from typing import Any, Callable, List
from tinynet.tensor import Tensor


class Context:
    def __init__(self, op_fn: Callable, *tensors: List[Tensor]) -> None:
        self.op_fn = op_fn
        self.parents = tensors
        self.saved_tensors = []

    def save_for_backward(self, *tensors: List[Tensor]) -> None:
        self.saved_tensors.extend(*tensors)


class Function:
    def __init__(self):
        cls = self.__class__
        raise RuntimeError(
            f"{cls} should not be instantiated. "
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
