import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

def get_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load the training and test datasets
    full_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    full_test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

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

    def get_tasks_zipped(self, batch_size, shuffle_train=False):
        # zips the train and test loaders
        return list(zip(self.get_train_loaders(batch_size, shuffle=shuffle_train), self.get_test_loaders(batch_size)))

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
