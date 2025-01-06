import numpy as np
import torch


class SimpleReplay:
    def __init__(self, samples_per_class=50, batch_size=4):
        self.size = samples_per_class
        self.memory = {}
        self.indices = {}
        self.batch_size = batch_size

    def add_by_batch(self, x, y):
        for i in range(len(x)):
            self.add(x[i], y[i])

    def add_by_loader(self, loader):
        for inputs, labels in loader:
            self.add_by_batch(inputs, labels)

    def add(self, x, y):
        y = int(y.detach().cpu().numpy())

        if y not in self.memory:
            self.memory[y] = []
            self.indices[y] = 0
        self.indices[y] += 1
        if len(self.memory[y]) < self.size:
            self.memory[y].append(x)
        else:
            # Perform reservoir sampling
            r = np.random.randint(0, self.indices[y])

            if r < self.size:
                self.memory[y][r] = x

    def sample_batch(self):
        if len(self.memory.keys()) == 0:
            return None, None

        x = []
        y = []
        for i in range(self.batch_size):
            label = np.random.choice(list(self.memory.keys()))
            x.append(self.memory[label][np.random.randint(0, len(self.memory[label]))])
            y.append(label)
        return torch.stack(x), torch.tensor(y)
