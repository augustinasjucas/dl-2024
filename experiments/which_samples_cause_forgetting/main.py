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
import os
import experiments.which_samples_cause_forgetting.models as models


def train_first_task_model(loader, model_type=models.CIFAR_CNN_1, device='cuda'):

    # shuffle the loader
    loader = DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=True)
    model = model_type()
    train(model, loader, optim.Adam(model.parameters(), lr=0.002), epochs=90, log_rate=1, device=device, dont_log=True)
    return model


def perform_single_step(model, x, y, lr=0.002, device="cuda"):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    outputs = model(x)
    loss = nn.CrossEntropyLoss()(outputs, y)
    optimizer.zero_grad()
    loss.backward()

    ret = model.get_gradient_norms_layerwise()

    optimizer.step()
    return ret

def process_single_element(idx, x, y, model_copy, first_task_loader, device='cuda', step_lr=0.001):
    layer_norms = perform_single_step(model_copy, x, y, device=device, lr=step_lr)
    new_test_accuracy, new_test_loss = test(model_copy, first_task_loader, device=device, move_to_device=False)        
    return idx, new_test_accuracy, new_test_loss, layer_norms


def get_scores_for_elements(model_type, task1_train_loader, task1_test_loader, task2_train_loader, n_repeats, step_lr=0.001, device='cuda', outpath="results"):

    os.makedirs(f"{outpath}", exist_ok=True)
    os.makedirs(f"{outpath}/results", exist_ok=True)

    # Write to a file the parameters of the experiment
    with open(f"{outpath}/params_and_logs.txt", "w") as f:
        f.write(f"n_repeats: {n_repeats}\n")
        f.write(f"step_lr: {step_lr}\n")
        f.write(f"Output path: {outpath}\n")
        model = model_type()
        f.write(str(model) + "\n")
        
    for R in range(n_repeats):
        print(f"  Starting iteration {R}")
        
        # Get a random good model and ensure it's on CPU
        first_task_model = train_first_task_model(task1_train_loader, model_type, device=device)
        first_task_model = first_task_model.to(device)
        
        # Calculate test accuracy on the second task
        original_test_accuracy, original_test_loss = test(first_task_model, task1_test_loader, device=device)
        with open(f"{outpath}/params_and_logs.txt", "a") as f:
            f.write(f"Run {R}. After training on the first task:\n")
            f.write(f"  First task test accuracy: {original_test_accuracy}\n")
            f.write(f"  First task test loss: {original_test_loss}\n\n")

        # Initialize empty array for differences
        total_elements = len(task2_train_loader)
        diffs = [None] * total_elements
        all_layer_norms = [None] * total_elements

        for i, (x, y) in enumerate(task2_train_loader):
            first_task_model_copy = copy.deepcopy(first_task_model)
            _, new_acc, new_loss, layer_norms = process_single_element(i, x, y, first_task_model_copy, task1_test_loader, step_lr)
            diffs[i] = (new_acc - original_test_accuracy, new_loss - original_test_loss)
            all_layer_norms[i] = layer_norms

        # Save diffs to txt file
        np.savetxt(f"{outpath}/results/diffs_{R}.txt", diffs, fmt='%f')

        # save layer norms to txt file (it is a list of lists of lists of strings)
        with open(f"{outpath}/results/layer_norms_{R}.txt", "w") as f:
            for layer_norms in all_layer_norms:
                for (name, norm) in layer_norms:
                    f.write(f"{name} {norm} ")
                f.write("\n")

        # Clean up
        del first_task_model


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

import experiments.which_samples_cause_forgetting.models as models

def main():
    # Argparse: we get single argument: ID
    parser = argparse.ArgumentParser()
    parser.add_argument("outpath", type=str)
    args = parser.parse_args()
    outpath = args.outpath

    full_train_dataset_cifar100, full_test_dataset_cifar100 = get_cifar()

    all_models = [models.CIFAR_CNN_1, models.CIFAR_CNN_2, models.CIFAR_CNN_3, models.CIFAR_CNN_4, models.CIFAR_MLP_1, models.CIFAR_MLP_2, models.CIFAR_MLP_3]
    step_sizes = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    
    full_train_dataset_cifar100 = GPUDataset(full_train_dataset_cifar100)
    full_test_dataset_cifar100 = GPUDataset(full_test_dataset_cifar100)

    task = FiveImageTasksCifar(full_train_dataset_cifar100, full_test_dataset_cifar100)

    for model_type in all_models:
        for step_size in step_sizes:
            path = f"{outpath}/{model_type.__name__}_{step_size}"
            print("Doing", model_type.__name__, "with step size", step_size)
            get_scores_for_elements(model_type, task.get_train_loaders(512, shuffle=True)[0], task.get_test_loaders(512)[0], task.get_train_loaders(1)[1], outpath=path, n_repeats=2, step_lr=step_size)

if __name__ == "__main__":
    main()