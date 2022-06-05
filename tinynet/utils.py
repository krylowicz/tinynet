import numpy as np
from .tensor import Tensor


def to_one_hot(labels: Tensor, n_classes: int) -> Tensor:
    labels = labels.data.astype(int)
    one_hot = np.zeros((labels.shape[0], n_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1

    return Tensor(one_hot, requires_grad=True)
