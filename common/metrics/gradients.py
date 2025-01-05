import torch
import wandb
from common.metrics import Metric
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


class GradientAlignment(Metric):
    def __init__(
        self,
        task_train_loaders,  # List of DataLoader, one for each task
        criterion,  # e.g. torch.nn.CrossEntropyLoss()
        device="cuda",
        check_every=1,  # how often to check (every 'n' epochs)
        wandb_params: dict = None,
    ):
        """
        A metric that computes pairwise angles between gradients from each task.

        Args:
            task_train_loaders: list of train loaders, one per task.
            criterion: the loss function to use in computing gradients.
            device: "cuda" or "cpu".
            check_every: check frequency (every 'n' steps or epochs).
            by_epoch: if True, check every 'n' epochs, else every 'n' batches.
            wandb_params: parameters for logging to W&B.
        """
        super().__init__("Gradient Alignment", "")
        self.task_train_loaders = task_train_loaders
        self.criterion = criterion
        self.device = device
        self.check_every = check_every
        self.wandb_params = wandb_params

        self.gradients = None
        self.results = None

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

    def after_epoch(self, model, task_num, epoch_num):
        """
        Called after every epoch. We'll do gradient alignment check if
        by_epoch=True and (epoch_num % check_every == 0).
        """
        if (epoch_num + 1) % self.check_every == 0:
            # Save the current model state for safe reversion
            # so that these gradient computations don't affect the actual training.
            current_state = {k: v.clone() for k, v in model.state_dict().items()}

            # Put model in train mode, just in case
            model.train()

            # For each task, get random batch, compute gradient
            gradients_list = []
            for loader in self.task_train_loaders:
                x, y = self._get_random_batch(loader)
                g = self._compute_gradient(model, x, y)
                gradients_list.append(g.detach().cpu().numpy())

                # restore model weights after each gradient so we don't accumulate
                model.load_state_dict(current_state)

            new_gradients = np.array(gradients_list)
            new_gradients = np.expand_dims(new_gradients, axis=0)

            if self.gradients is not None:
                self.gradients = np.concatenate((self.gradients, new_gradients), axis=0)
            else:
                self.gradients = new_gradients

            # Finally, restore model to original state
            model.load_state_dict(current_state)

    def after_all_tasks(self, model, tasks):
        num_tasks = len(self.task_train_loaders)
        avg_similarities = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(num_tasks):
                similarities = cosine_similarity(self.gradients[:, i, :], self.gradients[:, j, :])
                if i == j:
                    # exclude self-similarity
                    mask = np.ones(similarities.shape, dtype=bool)
                    np.fill_diagonal(mask, 0)
                    avg_similarities[i][j] = np.mean(similarities[mask])
                else:
                    avg_similarities[i][j] = np.mean(similarities)

        self.results = avg_similarities

        fig, ax = plt.subplots(figsize=(4, 3), dpi=300)  # High DPI for clearer image
        cax = ax.matshow(
            avg_similarities, cmap="viridis", interpolation="none"
        )  # Default colormap optimized for matrix data
        fig.colorbar(cax)
        ax.set_title("Average cosine similarity of gradients between tasks")
        ax.set_xlabel("Task index")
        ax.set_ylabel("Task index")
        ax.set_xticks(range(num_tasks))
        ax.set_yticks(range(num_tasks))

        plt.savefig("data/gradient_similarities.png", dpi=300)

        heatmap_logger = wandb.init(**self.wandb_params, name="gradient-alignment")
        wandb.log({"metrics-gradients/heatmap": wandb.Image("data/gradient_similarities.png")})
        heatmap_logger.finish()

        similarities_evolution = []
        for i in range(self.gradients.shape[0]):
            similarities = cosine_similarity(self.gradients[i, :, :])
            similarities_evolution.append(similarities)

        for i in range(num_tasks):
            # Initialize a new W&B run for each task
            taskwise_training_logger = wandb.init(**self.wandb_params, name=f"task-{i}")

            # Define a global step metric used by all metrics
            global_step_name = "metrics-gradients/measurement-step"
            wandb.define_metric(global_step_name)
            for j in range(num_tasks):
                metric_name = f"metrics-gradients/task-{j}"
                wandb.define_metric(metric_name, step_metric=global_step_name)

            # Log data
            for k in range(self.gradients.shape[0]):  # Use the outer loop for the global step
                for j in range(num_tasks):
                    if i != j:
                        metric_name = f"metrics-gradients/task-{j}"
                        log_data = {
                            metric_name: similarities_evolution[k][i, j],
                            global_step_name: k,  # Log the global step for this metric
                        }
                        wandb.log(log_data)

            # Finish the run after all logging is complete
            taskwise_training_logger.finish()

    def produce_result(self):
        """
        Return all the angles we collected for post-processing or printing.
        """
        return self.results

    def before_task(self, model, task_num, train_loader, test_loader):
        pass

    def after_task(self, model, task_num, train_loader, test_loader):
        pass
