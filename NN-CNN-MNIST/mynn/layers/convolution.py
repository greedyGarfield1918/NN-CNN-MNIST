import numpy as np
from .base import Layer


class conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 initialize_method=np.random.normal, weight_decay=False,
                 weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(1, out_channels))
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.cols = None

        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def _im2col(self, X):
        batch, in_c, H, W = X.shape
        k = self.kernel_size
        s = self.stride
        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1

        cols = np.zeros((batch, out_h * out_w, in_c * k * k))
        idx = 0
        for i in range(out_h):
            for j in range(out_w):
                patch = X[:, :, i * s:i * s + k, j * s:j * s + k]
                cols[:, idx, :] = patch.reshape(batch, -1)
                idx += 1
        return cols, out_h, out_w

    def forward(self, X):
        self.input = X
        batch = X.shape[0]
        cols, out_h, out_w = self._im2col(X)
        self.cols = cols

        W_col = self.W.reshape(self.out_channels, -1)
        out = cols @ W_col.T
        out = out.reshape(batch, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        out += self.b.reshape(1, self.out_channels, 1, 1)
        return out

    def backward(self, grads):
        batch = grads.shape[0]
        out_h, out_w = grads.shape[2], grads.shape[3]
        k = self.kernel_size
        s = self.stride

        # grad_W
        grads_2d = grads.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        cols_2d = self.cols.reshape(-1, self.in_channels * k * k)
        self.grads['W'] = (cols_2d.T @ grads_2d).T.reshape(self.W.shape)

        # grad_b
        self.grads['b'] = np.sum(grads_2d, axis=0).reshape(1, self.out_channels)

        # grad_input (col2im)
        W_col = self.W.reshape(self.out_channels, -1)
        grad_cols = grads_2d @ W_col
        grad_cols = grad_cols.reshape(batch, out_h, out_w, self.in_channels, k, k)

        grad_input = np.zeros_like(self.input)
        idx = 0
        for i in range(out_h):
            for j in range(out_w):
                patch_grad = grad_cols[:, i, j].reshape(batch, self.in_channels, k, k)
                grad_input[:, :, i * s:i * s + k, j * s:j * s + k] += patch_grad

        return grad_input

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}
