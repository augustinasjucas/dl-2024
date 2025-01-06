import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Download and load the training and test datasets
    full_train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    full_test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    return full_train_dataset, full_test_dataset


def get_cifar(version="10"):
    """
    Args:
        version (str): '10' for CIFAR-10 or '100' for 'CIFAR-100'
    """
    if version == "10":
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2470, 0.2435, 0.2616])

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        full_train_dataset = datasets.CIFAR10(
            root="data/", train=True, download=True, transform=transform
        )
        full_test_dataset = datasets.CIFAR10(
            root="data/", train=False, download=True, transform=transform
        )
    elif version == "100":
        mean = torch.tensor([0.5071, 0.4867, 0.4408])
        std = torch.tensor([0.2675, 0.2565, 0.2761])

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        # Download and load CIFAR-100
        full_train_dataset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        full_test_dataset = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise Exception("Unrecognized version: Pick CIFAR 10 or 100")

    return full_train_dataset, full_test_dataset


# Define a class for storing tasks split. It stores training dataset and validation dataset.
# The initializer takes the training dataset and validation dataset as input and
# images of a single digit. Same for validation dataset. Implemeting classes need to implement the split_dataset method.
class Tasks:
    def __init__(self, train_dataset, test_dataset):
        self.names = []
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_datasets = self.split_dataset(train_dataset)
        self.test_datasets = self.split_dataset(test_dataset)

    # Methods for getting train and test loaders
    def get_train_loaders(self, batch_size, shuffle=False):
        train_loaders = []
        for dataset in self.train_datasets:
            train_loaders.append(
                DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            )
        return train_loaders

    def get_test_loaders(self, batch_size):
        test_loaders = []
        for dataset in self.test_datasets:
            test_loaders.append(
                DataLoader(dataset, batch_size=batch_size, shuffle=False)
            )
        return test_loaders

    def get_tasks_zipped(self, batch_size):
        # zips the train and test loaders
        return list(
            zip(self.get_train_loaders(batch_size), self.get_test_loaders(batch_size))
        )

    def get_task_name(self, i):
        return self.names[i]


# Splits the training dataset into 10 sub-datasets. Each sub-dataset contains one digit.
class SingleDigitTasks(Tasks):
    # Split the dataset into 10 sub-datasets. Each sub-dataset contains images of a single digit
    def split_dataset(self, dataset):
        datasets = []
        for i in range(10):
            indices = torch.where(dataset.targets == i)[0]
            datasets.append(torch.utils.data.Subset(dataset, indices))
            self.names.append(f"Digit {i}")
        return datasets


class TwoDigitTasks(Tasks):
    # Split the dataset into 10 sub-datasets. Each sub-dataset contains images of a two digits: [0, 1], [2, 3], ..., [8, 9]
    def split_dataset(self, dataset):
        datasets = []
        self.names = []
        for i in range(0, 10, 2):
            indices = torch.where((dataset.targets == i) | (dataset.targets == i + 1))[
                0
            ]
            datasets.append(torch.utils.data.Subset(dataset, indices))
            self.names.append(f"Digits {i} and {i+1}")
        return datasets


class FiveDigitTasks(Tasks):
    # Split the dataset into 10 sub-datasets. Each sub-dataset contains images of a five digits: [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
    def split_dataset(self, dataset):
        datasets = []
        indices = torch.where((dataset.targets < 5))[0]
        datasets.append(torch.utils.data.Subset(dataset, indices))
        indices = torch.where((dataset.targets >= 5))[0]
        datasets.append(torch.utils.data.Subset(dataset, indices))
        self.names = ["Digits 0-4", "Digits 5-9"]
        return datasets


class FiveImageTasksCifar(Tasks):
    # Split the dataset into 100/5 = 20 sub-datasets. Each sub-dataset contains images of a five classes
    def split_dataset(self, dataset):
        datasets = []
        self.names = []
        for i in range(0, 100, 5):
            indices = torch.where(
                (torch.tensor(dataset.targets) >= i)
                & (torch.tensor(dataset.targets) < i + 5)
            )[0]
            datasets.append(torch.utils.data.Subset(dataset, indices))
            class_name = ", ".join([dataset.classes[j] for j in range(i, i + 5)])
            self.names.append(class_name)
        return datasets


class TenImageTasksCifar(Tasks):
    # Split the dataset into 100/5 = 20 sub-datasets. Each sub-dataset contains images of a five classes
    def split_dataset(self, dataset):
        datasets = []
        self.names = []
        for i in range(0, 100, 10):
            indices = torch.where(
                (torch.tensor(dataset.targets) >= i)
                & (torch.tensor(dataset.targets) < i + 10)
            )[0]
            datasets.append(torch.utils.data.Subset(dataset, indices))
            class_name = ", ".join([dataset.classes[j] for j in range(i, i + 10)])
            self.names.append(class_name)
        return datasets


class TwentyImageTasksCifar(Tasks):
    # Split the dataset into 100/5 = 20 sub-datasets. Each sub-dataset contains images of a five classes
    def split_dataset(self, dataset):
        datasets = []
        self.names = []
        for i in range(0, 100, 20):
            indices = torch.where(
                (torch.tensor(dataset.targets) >= i)
                & (torch.tensor(dataset.targets) < i + 20)
            )[0]
            datasets.append(torch.utils.data.Subset(dataset, indices))
            class_name = ", ".join([dataset.classes[j] for j in range(i, i + 20)])
            self.names.append(class_name)
        return datasets


class FiftyImageTasksCifar(Tasks):
    # Split the dataset into 100/5 = 20 sub-datasets. Each sub-dataset contains images of a five classes
    def split_dataset(self, dataset):
        datasets = []
        self.names = []
        for i in range(0, 100, 50):
            indices = torch.where(
                (torch.tensor(dataset.targets) >= i)
                & (torch.tensor(dataset.targets) < i + 50)
            )[0]
            datasets.append(torch.utils.data.Subset(dataset, indices))
            class_name = ", ".join([dataset.classes[j] for j in range(i, i + 50)])
            self.names.append(class_name)
        return datasets


class Cifar10SampleExperiment(Tasks):
    """
    Creates two tasks from CIFAR-10:
    - Task 1: First 5 classes with n samples each
    - Task 2: Last 5 classes with n samples each
    where n = experiment_number * 500
    """

    def __init__(self, train_dataset, test_dataset, experiment_number):
        if not 1 <= experiment_number <= 10:
            raise ValueError("Experiment number must be between 1 and 10")
        self.samples_per_class = experiment_number * 500
        super().__init__(train_dataset, test_dataset)

    def split_dataset(self, dataset):
        datasets = []
        self.names = []

        # Get indices for each class
        class_indices = {i: [] for i in range(10)}
        for idx, label in enumerate(dataset.targets):
            class_indices[label].append(idx)

        # Create two tasks: classes 0-4 and 5-9
        task_splits = [(0, 5), (5, 10)]

        for start, end in task_splits:
            task_indices = []
            for class_idx in range(start, end):
                # Randomly sample the specified number of images per class
                available_indices = class_indices[class_idx]
                if len(available_indices) < self.samples_per_class:
                    raise ValueError(
                        f"Not enough samples for class {class_idx}. Requested {self.samples_per_class}, but only {len(available_indices)} available."
                    )
                sampled_indices = random.sample(
                    available_indices, self.samples_per_class
                )
                task_indices.extend(sampled_indices)

            datasets.append(Subset(dataset, task_indices))
            class_names = [dataset.classes[j] for j in range(start, end)]
            self.names.append(
                f"Classes {', '.join(class_names)} ({self.samples_per_class} samples each)"
            )

        return datasets


class Cifar100ClassExperiment(Tasks):
    """
    Creates two tasks from CIFAR-100:
    - Task 1: First n classes
    - Task 2: Next n classes
    where n = experiment_number * 5
    """

    def __init__(self, train_dataset, test_dataset, experiment_number):
        if not 1 <= experiment_number <= 10:
            raise ValueError("Experiment number must be between 1 and 10")
        self.classes_per_task = experiment_number * 5
        super().__init__(train_dataset, test_dataset)

    def split_dataset(self, dataset):
        datasets = []
        self.names = []

        # Create two tasks with specified number of classes each
        task_splits = [
            (0, self.classes_per_task),
            (self.classes_per_task, 2 * self.classes_per_task),
            (2 * self.classes_per_task, 3 * self.classes_per_task),
            (3 * self.classes_per_task, 4 * self.classes_per_task),
            (4 * self.classes_per_task, 5 * self.classes_per_task),
            (5 * self.classes_per_task, 6 * self.classes_per_task),
            (6 * self.classes_per_task, 7 * self.classes_per_task),
            (7 * self.classes_per_task, 8 * self.classes_per_task),
            (8 * self.classes_per_task, 9 * self.classes_per_task),
            (9 * self.classes_per_task, 10 * self.classes_per_task),
            (10 * self.classes_per_task, 11 * self.classes_per_task),
            (11 * self.classes_per_task, 12 * self.classes_per_task),
            (12 * self.classes_per_task, 13 * self.classes_per_task),
            (13 * self.classes_per_task, 14 * self.classes_per_task),
            (14 * self.classes_per_task, 15 * self.classes_per_task),
            (15 * self.classes_per_task, 16 * self.classes_per_task),
            (16 * self.classes_per_task, 17 * self.classes_per_task),
            (17 * self.classes_per_task, 18 * self.classes_per_task),
            (18 * self.classes_per_task, 19 * self.classes_per_task),
        ]

        for start, end in task_splits:
            indices = []
            for class_idx in range(start, end):
                class_indices = torch.where(torch.tensor(dataset.targets) == class_idx)[
                    0
                ]
                indices.extend(class_indices.tolist())

            datasets.append(Subset(dataset, indices))
            class_names = [dataset.classes[j] for j in range(start, end)]
            self.names.append(f"Classes {', '.join(class_names)}")

        return datasets


def get_experiment_setup(dataset_type="cifar10", batch_size=256, experiment_number=1):
    """
    Returns data loaders for experiments with configurable size.

    For CIFAR-10:
    - Task 1: First 5 classes, (experiment_number * 500) samples each
    - Task 2: Last 5 classes, (experiment_number * 500) samples each

    For CIFAR-100:
    - Task 1: First (experiment_number * 5) classes (all samples)
    - Task 2: Next (experiment_number * 5) classes (all samples)

    Args:
        dataset_type: Either "cifar10" or "cifar100"
        batch_size: Batch size for the data loaders
        experiment_number: Number from 1-10 that determines experiment size

    Returns:
        List[Tuple[DataLoader, DataLoader]]: List of (train_loader, test_loader) pairs
    """
    if not 1 <= experiment_number <= 10:
        raise ValueError("Experiment number must be between 1 and 10")

    # Get the dataset
    dataset_version = "10" if dataset_type == "cifar10" else "100"
    full_train_dataset, full_test_dataset = get_cifar(dataset_version)

    if dataset_type == "cifar10":
        # Scale samples: experiment_number * 500 samples per class
        task = Cifar10SampleExperiment(
            full_train_dataset, full_test_dataset, experiment_number
        )
    else:  # cifar100
        # Scale classes: experiment_number * 5 classes per task
        task = Cifar100ClassExperiment(
            full_train_dataset, full_test_dataset, experiment_number
        )

    return task.get_tasks_zipped(batch_size)
