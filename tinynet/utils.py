import numpy as np

from tinynet.tensor import Tensor


def to_one_hot(labels: Tensor, n_classes: int) -> Tensor:
    labels = labels.data.astype(int)
    one_hot = np.zeros((labels.shape[0], n_classes))
    one_hot[np.arange(one_hot.shape[0]), labels] = 1

    return Tensor(one_hot, requires_grad=True)


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def fetch(url: str) -> np.ndarray:
        import requests
        from pathlib import Path
        from hashlib import md5
        from gzip import decompress

        file_path = Path("/tmp") / md5(url.encode("utf-8")).hexdigest()

        if Path.is_file(file_path):
            with open(file_path, "rb") as f:
                data = f.read()
        else:
            with open(file_path, "wb") as f:
                data = requests.get(url).content
                f.write(data)

        return np.frombuffer(decompress(data), dtype=np.uint8).copy()

    X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    return X_train, y_train, X_test, y_test
