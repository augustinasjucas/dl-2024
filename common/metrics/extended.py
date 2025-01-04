import torch
import wandb
from common.metrics import Metric
from torch.utils.data import DataLoader
import numpy as np


class ExtendedGradientAnalysis(Metric):
    def __init__(
        self,
        device="cuda",
        criterion=None,  # e.g., torch.nn.CrossEntropyLoss()
        tasks_zipped=None,  # a list of (train_loader, test_loader)
        log_to_wandb=True,
    ):
        """
        Tracks various gradient-related statistics at the start of each task:
          - Overall gradient magnitude normalized by parameter norm.
          - Per-layer gradient norms.
          - Gradient w.r.t input (robustness measure).

        Args:
            device: "cuda" or "cpu"
            criterion: the loss function (e.g., nn.CrossEntropyLoss)
            tasks_zipped: list of (train_loader, test_loader) for each task
            log_to_wandb: if True, logs values in W&B (otherwise only stored in self.results)
        """
        super().__init__("Extended Gradient Analysis", "")
        self.device = device
        self.criterion = criterion
        self.tasks_zipped = tasks_zipped
        self.log_to_wandb = log_to_wandb
        self.results = []

    def _get_random_batch(self, loader: DataLoader):
        """
        Utility to grab a random batch from a DataLoader.
        """
        # Depending on dataset size, you may want a more robust approach,
        # but for simplicity:
        data_iter = iter(loader)
        x, y = next(data_iter)
        return x.to(self.device), y.to(self.device)

    def _measure_gradients(self, model, x, y):
        """
        1) Zero out grads.
        2) Make x require grad for input-grad measurement.
        3) Forward pass => compute loss => backward pass.
        4) Measure:
           - total grad norm vs total param norm
           - per-layer grad norms
           - input grad norms (robustness)
        """

        # Make sure we can compute grad w.r.t. input for robustness measure
        x.requires_grad_(True)

        # Zero existing gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Forward & backward
        out = model(x)
        loss = self.criterion(out, y)
        loss.backward()

        # ---- 1. Overall gradient magnitude (normalized) ----
        total_param_norm = 0.0
        total_grad_norm = 0.0

        # We'll also store per-layer grads
        layer_grad_norms = {}

        # Iterate over named params
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            p_norm = param.data.norm(2)
            g_norm = param.grad.data.norm(2)
            total_param_norm += p_norm.pow(2).item()
            total_grad_norm += g_norm.pow(2).item()
            layer_grad_norms[name] = g_norm.item()

        total_param_norm = total_param_norm**0.5
        total_grad_norm = total_grad_norm**0.5
        if total_param_norm > 0.0:
            grad_ratio = total_grad_norm / total_param_norm
        else:
            grad_ratio = 0.0

        # ---- 2. Input (x) gradient norm (robustness measure) ----
        # For classification tasks, you might average this across the batch
        if x.grad is not None:
            # shape is [batch_size, channels, height, width], e.g. for images
            input_grad_norm = x.grad.view(x.size(0), -1).norm(2, dim=1).mean().item()
        else:
            input_grad_norm = 0.0

        return {
            "loss": loss.item(),
            "total_grad_norm": total_grad_norm,
            "total_param_norm": total_param_norm,
            "grad_ratio": grad_ratio,  # grad / param
            "layer_grad_norms": layer_grad_norms,
            "input_grad_norm": input_grad_norm,
        }

    def before_task(self, model, task_num, train_loader, test_loader):
        """
        Called before training each new task. We'll measure gradient info using
        a single random batch from the new task's training set.
        """
        x, y = self._get_random_batch(train_loader)
        metrics = self._measure_gradients(model, x, y)

        # Collect the data in self.results
        task_info = {
            "task_num": task_num,
            "loss": metrics["loss"],
            "total_grad_norm": metrics["total_grad_norm"],
            "param_norm": metrics["total_param_norm"],
            "grad_ratio": metrics["grad_ratio"],
            "input_grad_norm": metrics["input_grad_norm"],
            "layer_grad_norms": metrics["layer_grad_norms"],
        }
        self.results.append(task_info)

        # Optionally log to W&B
        if self.log_to_wandb:
            wandb.log(
                {
                    f"extended_grad_analysis/task_{task_num}/loss": metrics["loss"],
                    f"extended_grad_analysis/task_{task_num}/total_grad_norm": metrics["total_grad_norm"],
                    f"extended_grad_analysis/task_{task_num}/param_norm": metrics["total_param_norm"],
                    f"extended_grad_analysis/task_{task_num}/grad_ratio": metrics["grad_ratio"],
                    f"extended_grad_analysis/task_{task_num}/input_grad_norm": metrics["input_grad_norm"],
                }
            )

            # Also log per-layer grad norms, if desired
            for name, norm_val in metrics["layer_grad_norms"].items():
                wandb.log({f"extended_grad_analysis/task_{task_num}/grad_norm/{name}": norm_val})

        # After measuring, restore model to ensure no leftover grads
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

    # No-ops for the other hooks, unless you want something else
    def after_task(self, model, task_num, train_loader, test_loader):
        pass

    def after_batch(self, model, task_num, epoch_num, batch_num, batch_x, batch_y, batch_pred):
        pass

    def after_epoch(self, model, task_num, epoch_num):
        pass

    def after_all_tasks(self, model, tasks):
        pass

    def produce_result(self):
        """
        Return the list of dicts (one per task) with all measurements.
        """
        return self.results
