import numpy as np
from .sgd import Optimizer


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self.v = {}

    def step(self):
        for idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                for key in layer.params.keys():
                    v_key = (idx, key)
                    if v_key not in self.v:
                        self.v[v_key] = np.zeros_like(layer.params[key])
                    self.v[v_key] = self.mu * self.v[v_key] + self.init_lr * layer.grads[key]
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.v[v_key]
