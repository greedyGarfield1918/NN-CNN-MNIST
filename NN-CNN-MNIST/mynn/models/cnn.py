import numpy as np
from ..layers import Layer, conv2D, Linear, ReLU, Flatten
import pickle


def he_normal(size):
    fan_in = np.prod(size[1:])
    std = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, std, size=size)


class Model_CNN(Layer):
    def __init__(self, in_channels=1, num_classes=10):
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.layers = [
            conv2D(in_channels, 8, kernel_size=3, initialize_method=he_normal),
            ReLU(),
            conv2D(8, 16, kernel_size=3, initialize_method=he_normal),
            ReLU(),
            Flatten(),
            Linear(16 * 24 * 24, num_classes, initialize_method=he_normal),
        ]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            params = pickle.load(f)
        for i, layer in enumerate(self.layers):
            if layer.optimizable and f'layer_{i}' in params:
                layer.W = params[f'layer_{i}']['W']
                layer.b = params[f'layer_{i}']['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b

    def save_model(self, save_path):
        param_dict = {}
        for i, layer in enumerate(self.layers):
            if layer.optimizable:
                param_dict[f'layer_{i}'] = {
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                }
        with open(save_path, 'wb') as f:
            pickle.dump(param_dict, f)
