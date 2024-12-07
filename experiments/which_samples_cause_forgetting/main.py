from common.datasets import get_cifar, FiveDigitTasks
from common.models import MLP
from common.routines import train, test
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from common.datasets import FiveImageTasksCifar
import copy
from tqdm import tqdm
import argparse
   
class CIFAR_CNN(nn.Module):
    def __init__(self):
        super(CIFAR_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 3 input channel, 16 output channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 16 input, 32 output
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32 input, 64 output
        self.fc1 = nn.Linear(64 * 4 * 4, 64)  # Flattened size is 64 * 4 * 4
        self.fc2 = nn.Linear(64, 100)  # 100 classes for CIFAR
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # First conv + relu + pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Second conv + relu + pooling
        x = self.pool(torch.relu(self.conv3(x)))  # Third conv + relu + pooling
        x = x.view(-1, 64 * 4 * 4)  # Flatten
        x = torch.relu(self.fc1(x))  # Fully connected layer 1
        x = self.dropout(x)          # Dropout
        x = self.fc2(x)              # Fully connected layer 2 (output layer)
        return x

def train_first_task_model(loader, model_type=CIFAR_CNN, device='cuda'):

    # shuffle the loader
    loader = DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=True)
    model = model_type()
    train(model, loader, optim.Adam(model.parameters(), lr=0.002), epochs=90, log_rate=1, device=device)
    return model


def perform_single_step(model, x, y, lr=0.002, device="cuda"):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    outputs = model(x)
    loss = nn.CrossEntropyLoss()(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def process_single_element(idx, x, y, model_copy, first_task_loader, device='cuda'):
    perform_single_step(model_copy, x, y, device=device)
    new_test_accuracy, new_test_loss = test(model_copy, first_task_loader, device=device, move_to_device=False)        
    return idx, new_test_accuracy, new_test_loss


def get_scores_for_elements(task1_train_loader, task1_test_loader, task2_train_loader, n_repeats, device='cuda', ID="default"):
    ret = []
    
    for R in range(n_repeats):
        print(f"Starting iteration {R}")
        
        # Get a random good model and ensure it's on CPU
        first_task_model = train_first_task_model(task1_train_loader, device=device)
        first_task_model = first_task_model.to(device)
        
        # Calculate test accuracy on the second task
        original_test_accuracy, original_test_loss = test(first_task_model, task1_test_loader, device=device)
        print(f"Primary test accuracy on the first task is {original_test_accuracy}, and loss is {original_test_loss}")
                
        # Initialize empty array for differences
        total_elements = len(task2_train_loader)
        diffs = [None] * total_elements

        for i, (x, y) in enumerate(task2_train_loader):
            first_task_model_copy = copy.deepcopy(first_task_model)
            _, new_acc, new_loss = process_single_element(i, x, y, first_task_model_copy, task1_test_loader)            
            diffs[i] = (new_acc - original_test_accuracy, new_loss - original_test_loss)

        
        # Save diffs to txt file
        np.savetxt(f"results/{ID}_results_{R}.txt", diffs, fmt='%f')

        ret.append(diffs)
        
        # Clean up
        del first_task_model
        
    return ret



class GPUDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.items = []
        self.targets = []
        self.classes = dataset.classes
        for x, y in tqdm(dataset, "Moving dataset to GPU"):
            self.items.append((x.to("cuda"), torch.tensor(y).to("cuda")))
            self.targets.append(y)
    
    def __getitem__(self, idx):
        return self.items[idx]
    
    def __len__(self):
        return len(self.items)


def main():

    # Argparse: we get single argument: ID
    parser = argparse.ArgumentParser()
    parser.add_argument("ID", type=str)
    args = parser.parse_args()
    ID = args.ID

    full_train_dataset_cifar100, full_test_dataset_cifar100 = get_cifar()
    full_train_dataset_cifar100 = GPUDataset(full_train_dataset_cifar100)
    full_test_dataset_cifar100 = GPUDataset(full_test_dataset_cifar100)

    task = FiveImageTasksCifar(full_train_dataset_cifar100, full_test_dataset_cifar100)

    get_scores_for_elements(task.get_train_loaders(512, shuffle=True)[0], task.get_test_loaders(512)[0], task.get_train_loaders(1)[1], 3, ID=ID)

if __name__ == "__main__":
    main()