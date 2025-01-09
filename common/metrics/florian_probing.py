import copy
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

import wandb
from common.metrics import Metric
from common.utils import simple_test, simple_train


class CIFARDedupConcatDataset(Dataset):
    """
    A dataset that concatenates multiple CIFAR datasets while removing duplicates.
    Specifically handles CIFAR's structure and normalization.
    """

    def __init__(self, datasets: List[Dataset], dataset_type: str = "cifar10"):
        self.datasets = datasets
        self.dataset_type = dataset_type
        self.indices = []
        self._deduplicate()

        # Store normalization parameters
        if dataset_type == "cifar10":
            self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
            self.std = torch.tensor([0.2470, 0.2435, 0.2616])
        else:  # cifar100
            self.mean = torch.tensor([0.5071, 0.4867, 0.4408])
            self.std = torch.tensor([0.2675, 0.2565, 0.2761])

    def _image_to_hash(self, image):
        """Convert image tensor to a hashable format"""
        # Denormalize the image first to ensure consistent comparison
        image = image.cpu().numpy()
        return hash(image.tobytes())

    def _deduplicate(self):
        """Build index of unique samples across all datasets"""
        seen_samples = set()

        for dataset_idx, dataset in enumerate(self.datasets):
            if isinstance(dataset, torch.utils.data.Subset):
                parent_dataset = dataset.dataset
                indices = dataset.indices
            else:
                parent_dataset = dataset
                indices = range(len(dataset))

            for idx in indices:
                image, label = parent_dataset[idx]
                image_hash = self._image_to_hash(image)

                if image_hash not in seen_samples:
                    seen_samples.add(image_hash)
                    self.indices.append((dataset_idx, idx))

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.indices[idx]
        dataset = self.datasets[dataset_idx]

        if isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset[dataset.indices[sample_idx]]
        return dataset[sample_idx]

    def __len__(self):
        return len(self.indices)


def merge_cifar_dataloaders_dedup(
    dataloaders: List[DataLoader],
    batch_size: int,
    dataset_type: str = "cifar10",
    shuffle: bool = True,
) -> DataLoader:
    """
    Merge multiple CIFAR dataloaders while removing duplicates.

    Args:
        dataloaders: List of DataLoader objects to merge
        batch_size: Batch size for the merged dataloader
        dataset_type: Either "cifar10" or "cifar100"
        shuffle: Whether to shuffle the merged dataset

    Returns:
        A new DataLoader containing unique samples from all input dataloaders
    """
    datasets = [loader.dataset for loader in dataloaders]
    merged_dataset = CIFARDedupConcatDataset(datasets, dataset_type)

    return DataLoader(
        merged_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=dataloaders[0].num_workers
        if hasattr(dataloaders[0], "num_workers")
        else 0,
    )


class FlorianProbing(Metric):
    def __init__(
        self,
        layers_order: List[Tuple[List[torch.nn.Module], str]],
        all_layers: List[torch.nn.Module],
        optimizer: Tuple[torch.optim.Optimizer, dict],
        criterion: torch.nn.Module,
        epochs: int,
        batch_size: int = 256,
        device: str = "cuda",
        wandb_params: dict = None,
        sweepy_logging: bool = False,
    ):
        """
        ==== NOTE: it is CRUCIAL that you dont change the "model" class inside. For instance, do not change the code of this class to model = model.to(device).
                   Do all of the model.to(device) before passing the model to this function. That is because the layers_order must retain the references to the original model ====

        Args:
            layers_order:   the ith element of this list contains the NEW layers that should be added to the freeze list after the ith layer.
                            The reson why it is a list of lists instead of list of modules is that maybe for some models you will want to freeze
                            a few layers at once (maybe the bias and the weights are separate layers, for example). Also, every element of the list
                            has a name, which is the name of the layer that was just frozen.
            all_layers:     a list of all layers of the model that have parameters. This is needed so that I can reinitialize them easily
            optimizer: the optimizer to use for training the model on all data from all tasks. It is a tuple of the optimizer type and its parameters.
            criterion: the loss function to use
            epochs: the number of epochs to train the model on all data from all tasks
            batch_size: the batch size to use for constructing a dataset of ALL tasks
            wandb_params: the parameters to pass to wandb.init
            sweepy_logging: if True, will log the results in such way that everything will be a SINGLE run. Results in not-so-nice graphs, but easy to run sweeps
        """
        self.layers_order = layers_order
        self.all_layers = all_layers
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.device = device
        self.criterion = criterion
        self.epochs = epochs
        self.wandb_params = wandb_params
        self.sweepy_logging = sweepy_logging

        super().__init__(
            "Florian Probing",
            """
            After training the model on all CL tasks in a row, this metrics does this:
                1. Iterates one by one over the layers of the model.
                    1. On the ith iteration, it sets the weights of all layers 0-i to the ones they had after training after training on all tasks
                    and the remaining layers are just reinitialized.
                    2. The first i layers are then frozen.
                    3. Then, this model is trained on ALL data from ALL tasks (not in a CL way, just normally).
                    4. The test data of all tasks is then passed through the model, and the loss and accuracy are computed.
                    5. This is the score that is returned for this layer.
                2. We return a list of scores, one for each layer.
            """,
        )
        self.results = []

    def after_all_tasks(self, model, tasks: List[Tuple[DataLoader, DataLoader]]):
        # Determine if we're using CIFAR-10 or CIFAR-100 based on the number of classes
        dataset_type = (
            "cifar100" if len(tasks[0][0].dataset.dataset.classes) == 100 else "cifar10"
        )

        # Create deduped train and test loaders excluding the last task
        full_train_loader = merge_cifar_dataloaders_dedup(
            [task[0] for task in tasks[:-1]],
            batch_size=self.batch_size,
            dataset_type=dataset_type,
            shuffle=True,
        )

        full_test_loader = merge_cifar_dataloaders_dedup(
            [task[1] for task in tasks[:-1]],
            batch_size=self.batch_size,
            dataset_type=dataset_type,
            shuffle=True,
        )

        all_layers_to_freeze = []

        for new_layers_to_freeze, name in self.layers_order:
            if self.sweepy_logging:
                sweepy_string = f"layer-{name}/"
            else:
                layerwise_training_logger = wandb.init(
                    **self.wandb_params, name=f"layer-{name}"
                )
                wandb.Settings(quiet=True)
                sweepy_string = ""

            all_layers_to_freeze.extend(new_layers_to_freeze)

            # Make a copy of the model for later reconstruction of "model"
            model_copy = copy.deepcopy(model)

            # first, freeze all needed layers
            for layer in all_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

            # then reset the parameters of all layers that are not frozen
            for layer in self.all_layers:
                if layer not in all_layers_to_freeze:
                    layer.reset_parameters()

            print("Before training, model is ", model)
            # print parameters and their requires_grad and first 5 numbers of (flattened) params
            for name, param in model.named_parameters():
                print("  ", name, param.requires_grad, param.flatten()[:5])

            print()

            # now train the model on all data from all tasks
            optimizer = self.optimizer[0](model.parameters(), **self.optimizer[1])
            simple_train(
                model,
                full_train_loader,
                optimizer,
                self.criterion,
                self.epochs,
                self.device,
                f"metrics-florian_probing/intermediate-training-results/{sweepy_string}",
                test_loader=full_test_loader,
            )

            loss, acc = simple_test(
                model, full_test_loader, self.criterion, self.device
            )
            loss_train, acc_train = simple_test(
                model, full_train_loader, self.criterion, self.device
            )

            # append the results
            self.results.append((name, acc, loss, acc_train, loss_train))

            wandb.log(
                {
                    f"metrics-florian_probing/final-results/{sweepy_string}test-accuracy": acc,
                    f"metrics-florian_probing/final-results/{sweepy_string}test-loss": loss,
                    f"metrics-florian_probing/final-results/{sweepy_string}train-accuracy": acc_train,
                    f"metrics-florian_probing/final-results/{sweepy_string}train-loss": loss_train,
                    f"metrics-florian_probing/final-results/{sweepy_string}layer-index": len(
                        self.results
                    )
                    - 1,
                }
            )

            if not self.sweepy_logging:
                layerwise_training_logger.finish()

            # now, fix "model": set all layers to requires_grad=True and set the weights from the copy
            for name, param in model.named_parameters():
                param.requires_grad = True
                param.data = model_copy.state_dict()[name]

    def produce_result(self):
        return self.results

    def after_task(self, model, task_num, train_loader, test_loader):
        pass

    def after_batch(
        self, model, task_num, epoch_num, batch_num, batch_x, batch_y, batch_pred
    ):
        pass

    def after_epoch(self, model, task_num, epoch_num):
        pass

    def before_task(self, model, task_num, train_loader, test_loader):
        pass

    def after_task(self, model, task_num, train_loader, test_loader):
        pass
