import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


# Define a function that takes a model, takes a dataloader and returns the accuracy of the model. The outputs of the model are a softmaxed vector!
def test(model, test_loader, device='cuda', move_to_device=True):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            if move_to_device:
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += nn.CrossEntropyLoss()(outputs, labels).item()
    return correct / total, total_loss / len(test_loader)


# Training loop
def train(model, train_loader, optimizer, loss_type=nn.CrossEntropyLoss(), epochs=2000, log_rate=100, 
          test_during_epoch=False, test_batch_freq=1, test_during_epoch_callback=None, device='cuda', 
          gradient_update_function=None, replay_strategy=None, move_to_device=True):
    
    model = model.to(device)

    model.train()

    # Calculate running loss and accruracy    
    running_loss = 0.0
    running_corrects = 0
    running_total = 0

    total_batch_number = 0

    for epoch in range(epochs):
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            
            # Move to gpu/cpu
            if move_to_device:
                inputs, labels = inputs.to(device), labels.to(device)

            # Append some past samples to the batch
            if replay_strategy is not None:
                replay_inputs, replay_labels = replay_strategy.sample_batch()
                if replay_inputs is not None:
                    replay_inputs, replay_labels = replay_inputs.to(device), replay_labels.to(device)
                    inputs = torch.cat([inputs, replay_inputs])
                    labels = torch.cat([labels, replay_labels])

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_type(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            if gradient_update_function is not None:
                gradient_update_function(model)

            optimizer.step()
            
            # Accumulate loss for reporting
            running_loss += loss.item()
            running_corrects += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
            running_total += labels.size(0)

            if test_during_epoch:
                if batch_idx % test_batch_freq == test_batch_freq - 1:
                    # print(f"After batch {batch_idx + 1} accuracies for previous tasks are:")
                    test_during_epoch_callback(model, total_batch_number)
                    # print()
            total_batch_number += 1
        
        # Print average loss every log_rate epochs
        if epoch % log_rate == log_rate - 1:
            print(f'    Epoch {epoch + 1}, Loss: {running_loss / log_rate}, Train accuracy: {running_corrects / running_total}')
            running_loss = 0.0
            running_corrects = 0
            running_total = 0


def train_on_tasks(model, tasks, epochs=4, log_rate=1, plot_best_worst=False, batch_size=64, test_batch_size=64*16, lr=0.001,
                   test_during_epoch=False, test_batch_freq=1, device='cuda', plot_showing_freq=20, gradient_update_function=None, weight_decay=0.0,
                   batch_evaluation_frequency=1, print_intermediate_testing=False, replay_strategy=None):
    n_tasks = len(tasks.train_datasets)

    validation_results_on_previous_tasks = []

    # Go over tasks in a row
    for i in range(n_tasks):
        
        validation_results_on_previous_tasks.append([])

        def plot_tests_on_previous_tasks():
            for j in range(i+1):
                # xs will be the batch numbers: 0, test_batch_freq, 2*test_batch_freq, ...
                batch_indices = [x[0] for x in validation_results_on_previous_tasks[-1]]
                accuracies = [x[1][j] for x in validation_results_on_previous_tasks[-1]]

                plt.plot(batch_indices, accuracies, label=f"Task {j} ({tasks.get_task_name(j)})")
            
            plt.title(f"Accuracy of previous tasks during training on task {i}")
            plt.xlabel("Batch number")
            plt.ylabel("Accuracy")
            plt.legend()
           
            # Plot how parameter norms change
            ax = plt.twinx()
            norm_values = np.array([x[2] for x in validation_results_on_previous_tasks[-1]])
            ax.plot(batch_indices, norm_values, label="Parameter norm", color='black', linestyle='dashed')
            ax.set_ylabel("Parameter norm")
            ax.legend()

            plt.show()


        def norm(model):
            norm = 0
            for param in model.parameters():
                norm += torch.norm(param).item()
            return norm

        # Define a lambda for testing on all previous tasks, and saving the results
        def test_on_previous_tasks(model, batch_number=None):
            if batch_number is not None:
                if batch_number % batch_evaluation_frequency != 0:
                    return
                
                if batch_number >= 1000 * batch_evaluation_frequency:
                    if batch_number % (1000 * batch_evaluation_frequency) != 0: # only take every 1000th batch
                        return
                elif batch_number > (130 + (i * 20)) * batch_evaluation_frequency:
                    if batch_number % (100 * batch_evaluation_frequency) != 0:
                        return

            results_for_this_batch = []
            for j in range(i+1):
                test_loader = tasks.get_test_loaders(test_batch_size)[j]
                accuracy = test(model, test_loader, device=device)
                results_for_this_batch.append(accuracy)

                if print_intermediate_testing:
                    print("    accuracy on task ", j, " is ", accuracy)
            
            if batch_number is not None:
                validation_results_on_previous_tasks[-1].append((batch_number, results_for_this_batch, norm(model)))

            if len(validation_results_on_previous_tasks[-1]) % plot_showing_freq == 0 and batch_number is not None:
                plot_tests_on_previous_tasks()

        # Get the data
        train_loader = tasks.get_train_loaders(batch_size)[i]
        test_loader = tasks.get_test_loaders(batch_size)[i]

        # Define an optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

        # If we want to test how accuracy of previous tasks falls during training, as more batches are added,
        # we will just call the test_on_previous_tasks function every test_batch_freq batches
        if test_during_epoch: test_during_epoch_args = {"test_during_epoch": i != 0, "test_batch_freq": test_batch_freq, "test_during_epoch_callback": test_on_previous_tasks}
        else: test_during_epoch_args = {}

        print(f"Training on task {i}")

        train(model, train_loader, optimizer, epochs=(epochs if type(epochs) is int else epochs[i]), log_rate=log_rate, 
              # in case we want to see how testing accuracy falls during training
              **test_during_epoch_args, device=device, gradient_update_function=gradient_update_function, replay_strategy=replay_strategy)
        
        if replay_strategy is not None:
            replay_strategy.add_by_loader(train_loader)
        
        if plot_best_worst:
            print(f"    After training on task {i}, the following are best and worst clasisfied images for this task:")
            plot_best_worst(model, test_loader)

        test_on_previous_tasks(model)

        if test_during_epoch and i != 0:
            # Plot (i+1) plots, each showing how accuracy of previous tasks changes during training
            plot_tests_on_previous_tasks()


        print("\n\n\n")
        print("========================================")