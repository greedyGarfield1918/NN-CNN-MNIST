from .step_lr import scheduler


class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones, gamma=0.5):
        super().__init__(optimizer)
        self.milestones = milestones
        self.gamma = gamma

    def step(self):
        self.step_count += 1
        if self.step_count in self.milestones:
            self.optimizer.init_lr *= self.gamma
