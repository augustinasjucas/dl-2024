import torch
import wandb
from common.metrics import Metric
from torch.utils.data import DataLoader
import numpy as np
import random


class GradientAlignmentMetric(Metric):
    def __init__(
        self,
        task_train_loaders,  # List of DataLoader, one for each task
        criterion,  # e.g. torch.nn.CrossEntropyLoss()
        device="cuda",
        check_every=1,  # how often to check (every 'n' epochs or steps)
        by_epoch=True,  # if True, do it after each epoch; if False, do it after each batch
    ):
        """
        A metric that computes pairwise angles between gradients from each task.

        Args:
            task_train_loaders: list of train loaders, one per task.
            criterion: the loss function to use in computing gradients.
            device: "cuda" or "cpu".
            check_every: check frequency (every 'n' steps or epochs).
            by_epoch: if True, check every 'n' epochs, else every 'n' batches.
        """
        super().__init__("Gradient Alignment", "")
        self.task_train_loaders = task_train_loaders
        self.criterion = criterion
        self.device = device
        self.check_every = check_every
        self.by_epoch = by_epoch

        # We'll store our results (angles) here
        self.results = []

    def _get_random_batch(self, loader):
        """
        Utility to grab a random batch from a DataLoader.
        """
        # There's more than one way to do this; for large datasets, you might
        # just do next(iter(loader)) or sample a random index. For simplicity:
        data_iter = iter(loader)
        x, y = next(data_iter)
        return x.to(self.device), y.to(self.device)

    def _compute_gradient(self, model, x, y):
        """
        Compute gradients for a single batch (x, y).
        Return flattened gradient vector.
        """
        # Zero out existing gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Forward + backward
        outputs = model(x)
        loss = self.criterion(outputs, y)
        loss.backward()

        # Flatten all parameter grads into a single 1D tensor
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        return torch.cat(grads, dim=0)

    def _compute_pairwise_angles(self, grad_list):
        """
        Given a list of gradient vectors [g1, g2, ..., gT],
        compute pairwise angles and return the average angle.
        """
        angles = []
        T = len(grad_list)
        for i in range(T):
            for j in range(i + 1, T):
                dot_ij = torch.dot(grad_list[i], grad_list[j])
                norm_i = grad_list[i].norm()
                norm_j = grad_list[j].norm()
                cosine_ij = (dot_ij / (norm_i * norm_j)).clamp(-1.0, 1.0)  # avoid floating errors
                angle_ij = torch.acos(cosine_ij).item()  # in radians
                angles.append(angle_ij)
        if len(angles) == 0:
            return 0.0
        return float(np.mean(angles))  # average angle in radians

    def after_batch(self, model, task_num, epoch_num, batch_num, batch_x, batch_y, batch_pred):
        """
        Called after every batch in training. We'll do gradient alignment check if
        by_epoch=False and (batch_num % check_every == 0).
        """
        if not self.by_epoch:
            if (batch_num + 1) % self.check_every == 0:  # +1 for 1-based
                self._do_measurement(model, epoch_num, batch_num)

    def after_epoch(self, model, task_num, epoch_num):
        """
        Called after every epoch. We'll do gradient alignment check if
        by_epoch=True and (epoch_num % check_every == 0).
        """
        if self.by_epoch:
            if (epoch_num + 1) % self.check_every == 0:  # +1 for 1-based
                # We only measure once per epoch across tasks.
                # If you want this done per-task, you can move this call to after_task
                self._do_measurement(model, epoch_num, None)

    def _do_measurement(self, model, epoch_num, batch_num):
        # Save the current model state for safe reversion
        # so that these gradient computations don't affect the actual training.
        current_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Put model in train mode, just in case
        model.train()

        # For each task, get random batch, compute gradient
        grad_list = []
        for loader in self.task_train_loaders:
            x, y = self._get_random_batch(loader)
            g = self._compute_gradient(model, x, y)
            grad_list.append(g.detach().clone())

            # restore model weights after each gradient so we don't accumulate
            model.load_state_dict(current_state)

        # Compute average pairwise angle
        avg_angle = self._compute_pairwise_angles(grad_list)

        # Store or log it
        # Log to W&B for convenience
        step_id = epoch_num if batch_num is None else batch_num
        wandb.log({"gradient_alignment/avg_angle": avg_angle, "gradient_alignment/epoch": epoch_num})
        self.results.append((epoch_num, batch_num, avg_angle))

        # Finally, restore model to original state
        model.load_state_dict(current_state)

    def produce_result(self):
        """
        Return all the angles we collected for post-processing or printing.
        """
        return self.results

    # The rest are no-ops, unless you want to do something else
    def before_task(self, model, task_num, train_loader, test_loader):
        pass

    def after_task(self, model, task_num, train_loader, test_loader):
        pass
