import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import os
import zipfile
import requests
from torchvision import datasets, transforms
import random

def get_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load the training and test datasets
    full_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    full_test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    return full_train_dataset, full_test_dataset

def download_and_extract_tiny_imagenet(dataset_dir):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(dataset_dir, "tiny-imagenet-200.zip")

    # Create dataset directory if it doesn't exist
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Download the dataset if the ZIP file doesn't exist
    if not os.path.exists(zip_path):
        print("Downloading Tiny ImageNet dataset...")
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Download complete.")

    # Extract the dataset if the folder isn't already present
    extract_dir = os.path.join(dataset_dir, "tiny-imagenet-200")
    if not os.path.exists(extract_dir):
        print("Extracting Tiny ImageNet dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")

def get_tiny_imagenet():
    dataset_dir = "./data"
    download_and_extract_tiny_imagenet(dataset_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # resize to 32 32
        transforms.Resize([32, 32])
    ])

    # Define paths for train and validation datasets
    train_dir = os.path.join(dataset_dir, "tiny-imagenet-200", "train")
    val_dir = os.path.join(dataset_dir, "tiny-imagenet-200", "val")

    # Load the datasets
    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    full_test_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

    return full_train_dataset, full_test_dataset

def get_SVHN(set_one_class=False, new_label=100):
    """
    Args:
        - set_one_class (bool): If true, all the labels get set to `new_label`
        - new_label (int): The label to assign all data if set_one_class is specified
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL image to Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Example normalization
    ])
    
    full_train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    full_test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)

    if set_one_class:
        full_train_dataset.labels = np.full_like(full_train_dataset.labels, new_label)
        full_test_dataset.labels = np.full_like(full_test_dataset.labels, new_label)
    
    return full_train_dataset, full_test_dataset


def get_cifar(version='10'):
    """
    Args:
        version (str): '10' for CIFAR-10 or '100' for 'CIFAR-100'
    """
    if version == '10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2470, 0.2435, 0.2616])

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        full_train_dataset = datasets.CIFAR10(root='data/', train=True, download=True, transform=transform)
        full_test_dataset = datasets.CIFAR10(root='data/', train=False, download=True, transform=transform)
    elif version == '100':
        mean = torch.tensor([0.5071, 0.4867, 0.4408])
        std = torch.tensor([0.2675, 0.2565, 0.2761])

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        # Download and load CIFAR-100
        full_train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        full_test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    else:
        raise Exception('Unrecognized version: Pick CIFAR 10 or 100')

    return full_train_dataset, full_test_dataset


# Define a class for storing tasks split. It stores training dataset and validation dataset.
# The initializer takes the training dataset and validation dataset as input and
# images of a single digit. Same for validation dataset. Implemeting classes need to implement the split_dataset method.
class Tasks():
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
            train_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))
        return train_loaders

    def get_test_loaders(self, batch_size):
        test_loaders = []
        for dataset in self.test_datasets:
            test_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=False))
        return test_loaders

    def get_tasks_zipped(self, batch_size):
        # zips the train and test loaders
        return list(zip(self.get_train_loaders(batch_size, shuffle=True), self.get_test_loaders(batch_size)))


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
            indices = torch.where((dataset.targets == i) | (dataset.targets == i+1))[0]
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
            indices = torch.where((torch.tensor(dataset.targets) >= i) & (torch.tensor(dataset.targets) < i+5))[0]
            datasets.append(torch.utils.data.Subset(dataset, indices))
            class_name = ", ".join([dataset.classes[j] for j in range(i, i+5)])
            self.names.append(class_name)
        return datasets


class FiveImageTasksCifar10(Tasks):
    # add an argument: max_elems_per_class
    def __init__(self, train_dataset, test_dataset, max_elems_per_class=100):
        self.max_elems_per_class = max_elems_per_class
        super().__init__(train_dataset, test_dataset)


    # Split the dataset into 100/5 = 20 sub-datasets. Each sub-dataset contains images of a five classes
    def split_dataset(self, dataset):
        datasets = []
        self.names = []
        for i in range(0, 10, 5):
            indices = []
            for j in range(i, i+5):
                indices.extend(torch.where(torch.tensor(dataset.targets) == j)[0][:self.max_elems_per_class])
            indices = torch.tensor(indices)
            datasets.append(torch.utils.data.Subset(dataset, indices))
            class_name = ", ".join([dataset.classes[j] for j in range(i, i+5)])
            self.names.append(class_name)
        return datasets

class FiveImageTasksCifarMaxElems(Tasks):
    def __init__(self, train_dataset, test_dataset, max_elems_per_class=100):
        self.max_elems_per_class = max_elems_per_class
        super().__init__(train_dataset, test_dataset)

    # Split the dataset into 100/5 = 20 sub-datasets. Each sub-dataset contains images of a five classes
    def split_dataset(self, dataset):
        datasets = []
        self.names = []
        for i in range(0, 100, 5):
            indices = []
            for j in range(i, i+5):
                indices.extend(torch.where(torch.tensor(dataset.targets) == j)[0][:self.max_elems_per_class])
            indices = torch.tensor(indices)
            datasets.append(torch.utils.data.Subset(dataset, indices))
            class_name = ", ".join([dataset.classes[j] for j in range(i, i+5)])
            self.names.append(class_name)
        return datasets


class TenImageTasksCifar(Tasks):
    # Split the dataset into 100/5 = 20 sub-datasets. Each sub-dataset contains images of a five classes
    def split_dataset(self, dataset):
        datasets = []
        self.names = []
        for i in range(0, 100, 10):
            indices = torch.where((torch.tensor(dataset.targets) >= i) & (torch.tensor(dataset.targets) < i+10))[0]
            datasets.append(torch.utils.data.Subset(dataset, indices))
            class_name = ", ".join([dataset.classes[j] for j in range(i, i+10)])
            self.names.append(class_name)
        return datasets


class TwentyImageTasksCifar(Tasks):
    # Split the dataset into 100/5 = 20 sub-datasets. Each sub-dataset contains images of a five classes
    def split_dataset(self, dataset):
        datasets = []
        self.names = []
        for i in range(0, 100, 20):
            indices = torch.where((torch.tensor(dataset.targets) >= i) & (torch.tensor(dataset.targets) < i+20))[0]
            datasets.append(torch.utils.data.Subset(dataset, indices))
            class_name = ", ".join([dataset.classes[j] for j in range(i, i+20)])
            self.names.append(class_name)
        return datasets

class FiftyImageTasksCifar(Tasks):
    # Split the dataset into 100/5 = 20 sub-datasets. Each sub-dataset contains images of a five classes
    def split_dataset(self, dataset):
        datasets = []
        self.names = []
        for i in range(0, 100, 50):
            indices = torch.where((torch.tensor(dataset.targets) >= i) & (torch.tensor(dataset.targets) < i+50))[0]
            datasets.append(torch.utils.data.Subset(dataset, indices))
            class_name = ", ".join([dataset.classes[j] for j in range(i, i+50)])
            self.names.append(class_name)
        return datasets

class OneImageTasksCifar(Tasks):
    def split_dataset(self, dataset):
        datasets = []
        self.names = []
        for i in range(0, 100, 1):
            indices = torch.where((torch.tensor(dataset.targets) >= i) & (torch.tensor(dataset.targets) < i+1))[0]
            datasets.append(torch.utils.data.Subset(dataset, indices))
            class_name = ", ".join([dataset.classes[j] for j in range(i, i+1)])
            self.names.append(class_name)
        return datasets

class TwoImageTasksCifar(Tasks):
    def split_dataset(self, dataset):
        datasets = []
        self.names = []
        for i in range(0, 100, 2):
            indices = torch.where((torch.tensor(dataset.targets) >= i) & (torch.tensor(dataset.targets) < i+2))[0]
            datasets.append(torch.utils.data.Subset(dataset, indices))
            class_name = ", ".join([dataset.classes[j] for j in range(i, i+2)])
            self.names.append(class_name)
        return datasets


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


class RandomClassTasksCifar(Tasks):
    def __init__(self, train_dataset, test_dataset, task_size, random_seed=None):
        # This ensures reproducibility if needed
        if random_seed is not None:
            random.seed(random_seed)

        self.task_size = task_size

        # Shuffle classes once here
        self.all_classes = list(range(100))  # CIFAR-100 classes: 0..99
        random.shuffle(self.all_classes)

        super().__init__(train_dataset, test_dataset)

    def split_dataset(self, dataset):
        datasets = []
        self.names = []

        # Use the same shuffled class ordering for train and test
        for start in range(0, 100, self.task_size):
            end = start + self.task_size
            class_chunk = self.all_classes[start:end]

            # Collect indices of all images in these classes
            chunk_indices = []
            for idx, target in enumerate(dataset.targets):
                # Make sure target is the same type as class_chunk
                if target in class_chunk:
                    chunk_indices.append(idx)

            subset = torch.utils.data.Subset(dataset, chunk_indices)
            datasets.append(subset)

            # Create a user-friendly name
            chunk_class_names = [dataset.classes[c] for c in class_chunk]
            self.names.append(", ".join(chunk_class_names))

        return datasets



class WhiteNoiseDataset(Dataset):
    def __init__(self, num_samples, image_size, num_classes):
        """
        Args:
            num_samples (int): Total number of samples in the dataset.
            image_size (tuple): Shape of the images, e.g., (3, 32, 32) for RGB images.
            num_classes (int): Number of classes for the labels.
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.data = torch.rand((num_samples, *image_size))  # Random white noise images
        self.targets = torch.randint(0, num_classes, (num_samples,))  # Random labels for classes
        self.classes = [f"Class {i}" for i in range(num_classes)]  # Class names

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def get_white_noise(num_samples=50000, num_classes=100, image_size=(3, 32, 32)):
    train_dataset = WhiteNoiseDataset(num_samples, image_size, num_classes)
    test_dataset = WhiteNoiseDataset(num_samples // 5, image_size, num_classes)
    return train_dataset, test_dataset
