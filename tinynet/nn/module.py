from __future__ import annotations

from typing import Any, Iterable

from tinynet.tensor import Tensor


class Module:
    def __init__(self) -> None:
        self._submodules: dict[str, Module] = {}
        self._parameters: dict[str, Tensor] = {}
        self.is_train: bool = False

    def train(self) -> None:
        self.is_train = True

    def eval(self) -> None:
        self.is_train = False

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses of Module must implement the forward method.")

    def parameters(self) -> Iterable[Tensor]:
        self._ensure_is_initialized()

        yield from self._parameters.values()
        for module in self._submodules.values():
            yield from module.parameters()

    def add_parameter(self, key: str, value: Tensor) -> None:
        self._ensure_is_initialized()
        self._parameters[key] = value

    def add_module(self, key: str, value: Module) -> None:
        self._ensure_is_initialized()
        self._submodules[key] = value

    def __setattr__(self, key, value) -> None:
        if isinstance(value, Tensor) and value.is_parameter:
            self.add_parameter(key, value)
        elif isinstance(value, Module):
            self.add_module(key, value)

        object.__setattr__(self, key, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def _ensure_is_initialized(self) -> None:
        if self.__dict__.get("_submodules") is None:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized. Did you run super().__init__()?")
