from .step_lr import scheduler


class ExponentialLR(scheduler):
    def __init__(self, optimizer, gamma=0.99):
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self):
        self.step_count += 1
        self.optimizer.init_lr *= self.gamma
