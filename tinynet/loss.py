from tinynet.tensor import Tensor
from tinynet.utils import to_one_hot


class Loss:
    def __init__(self) -> None:
        pass

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError("You must implement forward function for custom loss")


class CrossEntropyLoss(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, predicted: Tensor, target: Tensor) -> Tensor:
        """
        Computes the cross entropy loss between input and target

        predicted: (batch_size, n_classes) raw, un-normalized scores for each class
        target: range [0, n_classes)
        """
        batch_size, n_classes = predicted.shape

        predicted = predicted.logsoftmax()
        target = to_one_hot(target, n_classes)

        return (-target * predicted).mean()
