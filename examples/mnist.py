import numpy as np

import tinynet.nn as nn
from tinynet.tensor import Tensor
from tinynet.optim import SGD
from tinynet.loss import CrossEntropyLoss
from tinynet.utils import load_mnist

from tqdm import trange


BATCH_SIZE = 128
X_train, y_train, X_test, y_test = load_mnist()

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
optim = SGD(model.parameters(), lr=0.01)
criterion = CrossEntropyLoss()

model.train()
for _ in (t := trange(1000)):
    samp = np.random.randint(0, X_train.shape[0], size=BATCH_SIZE)

    X = Tensor(X_train[samp].reshape((-1, 28 * 28)))
    y = Tensor(y_train[samp])

    out = model.forward(X)

    loss = criterion(out, y)
    loss.backward()

    optim.zero_grad()
    optim.step()

    cat = np.argmax(out.data, axis=1)
    accuracy = (cat == y.data).mean()
    t.set_description(f"loss {loss.data[0, 0]:.4f} accuracy {accuracy:.4f}")


model.eval()
y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28 * 28))))
y_test_preds = np.argmax(y_test_preds_out.data, axis=1)
eval_acc = (y_test == y_test_preds).mean()

print(f"eval accuracy: {eval_acc}")
