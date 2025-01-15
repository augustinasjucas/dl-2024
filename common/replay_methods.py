import torch
import numpy as np

class SimpleReplay():
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


class LimitedReplay():
    def __init__(self, max_dataloaders=3, samples_per_class=50, batch_size=4):
        self.max_dataloaders = max_dataloaders
        self.samples_per_class = samples_per_class
        self.batch_size = batch_size
        self.memory = {}  # Stores data for each class
        self.dataloader_queue = []  # Keeps track of dataloaders
        
    def add_by_loader(self, loader):
        # Remove the oldest dataloader if max capacity is reached
        if len(self.dataloader_queue) >= self.max_dataloaders:
            oldest_loader_classes = self.dataloader_queue.pop(0)
            for cls in oldest_loader_classes:
                del self.memory[cls]
                print("deleting class", cls, "memory now has", len(self.memory), "keys", flush=True)
        
        # Store the classes in the current loader
        current_classes = []
        
        for inputs, labels in loader:
            for x, y in zip(inputs, labels):
                y = int(y.detach().cpu().numpy())
                if y not in self.memory:
                    self.memory[y] = []
                if len(self.memory[y]) < self.samples_per_class:
                    self.memory[y].append(x)
                else:
                    # Perform reservoir sampling
                    r = np.random.randint(0, self.samples_per_class)
                    self.memory[y][r] = x
                if y not in current_classes:
                    current_classes.append(y)
        
        self.dataloader_queue.append(current_classes)
    
    def sample_batch(self):
        if not self.memory:
            return None, None
        
        x, y = [], []
        for _ in range(self.batch_size):
            label = np.random.choice(list(self.memory.keys()))
            x.append(self.memory[label][np.random.randint(0, len(self.memory[label]))])
            y.append(label)
        
        return torch.stack(x), torch.tensor(y)
