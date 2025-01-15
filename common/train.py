import torch
from typing import List, Tuple

# import dataloader
from torch.utils.data import DataLoader
from common.utils import simple_test
import copy
from common.metrics import Metric
import wandb


class CLTraining:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Tuple[torch.optim.Optimizer, dict],
        criterion: torch.nn.Module,
        device: str,
        metrics: List[Metric],
        tasks: List[Tuple[DataLoader, DataLoader]],
        epochs: int,
        description: str = "No description",
        wandb_params: dict = None,
    ):
        """Creates a CL Training object. For now, the full purpose of this object is that you can call run() on it.
        So in principle, imagine that you would just call run() with all of these parameters.

        Args:
            model (torch.nn.Module): The model to train
            optimizer (torch.optim.Optimizer): The optimizer to use: a pair of the optimizer type and its parameters
            criterion (torch.nn.Module): The loss function to use
            device (str): The device to run the training on
            metrics (List[Metric]): A list of metrics to use
            tasks (List[Tuple[DataLoader, DataLoader]]): A list of tuples, where each tuple is a pair of train and test DataLoaders
            epochs (int): The number of epochs to train every task for
            description (str, optional): A description of this training run, used for better results, wandb. This could include
                                         stuff such as the model architecture, the tasks descriptions, etc.
            wandb_params (dict, optional): The parameters to pass to wandb.init
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.tasks = tasks
        self.epochs = epochs
        self.description = description
        self.wandb_params = wandb_params

        self.metrics = metrics

    def train(
        self, model, train_loader, optimizer_type, optimizer_parameters, criterion, epochs, task_index, metrics, device
    ):
        # Performs vanilla training on the model, and given data and all needed info

        # Copy the optimizer
        optimizer = optimizer_type(model.parameters(), **optimizer_parameters)

        model.train()

        for epoch_num in range(epochs):
            # model.eval()

            # Inform the metrics of this epoch
            for metric in metrics:
                metric.before_epoch(model, task_index, epoch_num)

            for batch_num, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                batch_pred = model(batch_x)
                loss = criterion(batch_pred, batch_y)

                # if batch_num == 0:
                #     print(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Inform the metrics of this batch
                for metric in metrics:
                    metric.after_batch(model, task_index, epoch_num, batch_num, batch_x, batch_y, batch_pred)

            # Inform the metrics of this epoch
            for metric in metrics:
                metric.after_epoch(model, task_index, epoch_num)

    def run(self):
        """
        Runs the continual learning training process.
            - First, goes over all tasks in order:
                - Trains the model on that tasks' train set
                - Moves on to the next task.
            - After all tasks are done, returns the model and the produce_result() of all metrics
            - During training of all tasks, logs everything to wandb
        """

        # Inform the model of the start of the training
        for metric in self.metrics:
            metric.before_all_tasks(self.model, self.tasks, self)

        # Go over all provided tasks
        for task_index, (train_loader, test_loader) in enumerate(self.tasks):
            # Inform the metrics of the start of this task
            for metric in self.metrics:
                metric.before_task(self.model, task_index, train_loader, test_loader)

            # Perform simple training
            self.train(
                self.model,
                train_loader,
                self.optimizer[0],
                self.optimizer[1],
                self.criterion,
                self.epochs,
                task_index,
                self.metrics,
                self.device,
            )

            # Inform the metrics of this task
            for metric in self.metrics:
                metric.after_task(self.model, task_index, train_loader, test_loader)

        # Inform the metrics that all tasks are done
        for metric in self.metrics:
            metric.after_all_tasks(self.model, self.tasks)

        return self.model, [metric.produce_result() for metric in self.metrics]
