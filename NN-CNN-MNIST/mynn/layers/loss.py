import numpy as np
from .base import Layer


class MultiCrossEntropyLoss(Layer):
    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        self.batch_size = predicts.shape[0]
        self.labels = labels

        # Softmax (numerically stable)
        x_max = np.max(predicts, axis=1, keepdims=True)
        x_exp = np.exp(predicts - x_max)
        self.probs = x_exp / np.sum(x_exp, axis=1, keepdims=True)

        # Cross-entropy loss
        loss = -np.mean(np.log(self.probs[np.arange(self.batch_size), labels] + 1e-12))
        return loss

    def backward(self):
        # Gradient of cross-entropy with softmax: probs - one_hot(labels)
        self.grads = self.probs.copy()
        self.grads[np.arange(self.batch_size), self.labels] -= 1
        self.grads /= self.batch_size
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self


class L2Regularization(Layer):
    pass


def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
