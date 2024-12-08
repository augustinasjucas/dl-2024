import argparse
import numpy as np
import os
from common.datasets import FiveImageTasksCifar, get_cifar
import matplotlib.pyplot as plt

def get_scores(filename):
    accuracies = []
    losses = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # Line contains two numbers separated by a space: diff in acc and loss
            acc, loss = line.split()
            accuracies.append(float(acc))
            losses.append(float(loss))

    return accuracies, losses

def main():
    # argparse: input is path to folder
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to folder with result txt files")

    args = parser.parse_args()

    # Go over all files in the folder
    path = args.path
    files = os.listdir(path)

    accuracies = []
    losses = []
    for file in files:
        if file.endswith(".txt"):
            accs, los = get_scores(os.path.join(path, file))
            accuracies.append(accs)
            losses.append(los)

    # Calculate average and std over runs: every file was a different run
    accuracies = np.array(accuracies)
    losses = np.array(losses)

    mean_accs = np.mean(accuracies, axis=0)
    std_accs = np.std(accuracies, axis=0)

    mean_losses = np.mean(losses, axis=0)
    std_losses = np.std(losses, axis=0)

    full_train_dataset_cifar100, full_test_dataset_cifar100 = get_cifar()
    task = FiveImageTasksCifar(full_train_dataset_cifar100, full_test_dataset_cifar100)

    # Get the train loader for the second task
    dataset = task.get_train_loaders(1)[1].dataset


    # Find the indiceces of top 25 and bottom 25 elements in accuracies
    indices = np.argsort(mean_accs)
    top_25 = indices[-36:][::-1]
    bottom_25 = indices[:36]

    print("mean accs", mean_accs[top_25])
    print("mean accs", mean_accs[bottom_25])

    # Plot the top 25 and bottom 25 elements: create two 5x5 grids

    # Top 25
    fig, axs1 = plt.subplots(6, 6, figsize=(7, 7))
    for i, ax in enumerate(axs1.flat):
        im = (dataset[top_25[i]][0].permute(1, 2, 0) + 1) / 2
        ax.imshow(im.cpu().numpy())
        ax.set_title(f"Acc diff: {mean_accs[top_25[i]]:.2f}, Loss diff: {mean_losses[top_25[i]]:.2f}", fontsize=5)
        ax.axis('off')
    plt.suptitle("Top 25 elements in accuracy")
    plt.subplots_adjust(top=0.92, bottom=0.02, left=0.0, right=1.0, hspace=0.25, wspace=0.0)

    # Bottom 25
    fig2, axs2 = plt.subplots(6, 6, figsize=(7, 7))
    for i, ax in enumerate(axs2.flat):
        im = (dataset[bottom_25[i]][0].permute(1, 2, 0) + 1) / 2
        ax.imshow(im.cpu().numpy())
        ax.set_title(f"Acc diff: {mean_accs[bottom_25[i]]:.2f}, Loss diff: {mean_losses[bottom_25[i]]:.2f}", fontsize=5)
        ax.axis('off')
    plt.suptitle("Bottom 25 elements in accuracy")
    plt.subplots_adjust(top=0.92, bottom=0.02, left=0.0, right=1.0, hspace=0.25, wspace=0.0)
    plt.show()



if __name__ == '__main__':
    main()