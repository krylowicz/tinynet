from tinynet.tensor import Tensor


class Loss:
    def __init__(self) -> None:
        pass

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError("You must implement forward function for custom loss")
