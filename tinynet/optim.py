from typing import Iterable

import numpy as np

from tinynet.tensor import Tensor


class Optimizer:
    def __init__(self, params: Iterable[Tensor]) -> None:
        self.params = list(params)

    def step(self) -> None:
        raise NotImplementedError("Subclass of Optimizer must implement step method.")

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad.zero_grad()


class SGD(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float = 0.01) -> None:
        super().__init__(params)
        self.lr = lr

    def step(self) -> None:
        for param in self.params:
            param.data -= self.lr * param.grad.data


class RMSProp(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.001,
        decay: float = 0.9,
        eps: float = 1e-8
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.decay = decay
        self.eps = eps

        self.v = [Tensor.zeros(p.shape) for p in self.params]

    def step(self) -> None:
        for i, param in enumerate(self.params):
            self.v[i].data = self.decay * self.v[i].data + (1 - self.decay) * (param.grad.data * param.grad.data)
            param.data -= self.lr * param.grad.data / (np.sqrt(self.v[i].data) + self.eps)


class Adam(Optimizer):
    def __init__(
            self,
            params: Iterable[Tensor],
            lr: float = 0.001,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.b1 = betas[0]
        self.b2 = betas[1]
        self.eps = eps
        self.t = 0

        self.m = [Tensor.zeros(p.shape) for p in self.params]
        self.v = [Tensor.zeros(p.shape) for p in self.params]

    def step(self) -> None:
        self.t += 1
        for i, param in enumerate(self.params):
            self.m[i].data = self.b1 * self.m[i].data + (1 - self.b1) * param.grad.data
            self.v[i].data = self.b2 * self.v[i].data + (1 - self.b2) * np.square(param.grad.data)
            m_hat = self.m[i].data / (1 - self.b1 ** self.t)
            v_hat = self.v[i].data / (1 - self.b2 ** self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
