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
        Additionally, it can measure the gradient magnitude at the start of each task
        to get a sense of transfer (possibly normalized by parameter norm).
        """
        super().__init__("Gradient Alignment", "")
        self.task_train_loaders = task_train_loaders
        self.criterion = criterion
        self.device = device
        self.check_every = check_every
        self.wandb_params = wandb_params

        self.gradients = None
        self.results = None

        # Store gradient magnitudes at the beginning of each task.
        self.initial_gradient_magnitudes_normalized = []
        self.initial_gradient_magnitudes = []

    def _get_random_batch(self, loader):
        """
        Utility to grab a random batch from a DataLoader.
        """
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
        (epoch_num + 1) % check_every == 0.
        """
        if (epoch_num + 1) % self.check_every == 0:
            # Save the current model state for safe reversion
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

    def before_task(self, model, task_num, train_loader, test_loader):
        """
        Called before each new task begins. We measure the gradient magnitude on
        this new task (using a random batch), normalized by the parameter norm.
        """
        # Save the current model state for safe reversion
        current_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.train()
        x, y = self._get_random_batch(train_loader)
        g = self._compute_gradient(model, x, y)

        # Compute norms
        with torch.no_grad():
            param_norm = 0.0
            for p in model.parameters():
                param_norm += p.data.norm() ** 2
            param_norm = param_norm.sqrt()

            grad_norm = g.norm()
            grad_norm_normalized = grad_norm / (param_norm + 1e-8)  # Avoid div by zero

        self.initial_gradient_magnitudes_normalized.append(grad_norm_normalized.item())
        self.initial_gradient_magnitudes.append(grad_norm.item())

        # Restore model weights
        model.load_state_dict(current_state)

    def after_task(self, model, task_num, train_loader, test_loader):
        pass

    def after_all_tasks(self, model, tasks):
        """
        Called once after all tasks are completed. We compute the average
        pairwise cosine similarities for each task pair. We also log the
        initial gradient magnitudes we recorded.
        """
        num_tasks = len(self.task_train_loaders)
        avg_similarities = np.zeros((num_tasks, num_tasks))

        # Compute average pairwise similarities
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

        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
        cax = ax.matshow(avg_similarities, cmap="viridis", interpolation="none")
        fig.colorbar(cax)
        ax.set_title("Average cosine similarity of gradients between tasks")
        ax.set_xlabel("Task index")
        ax.set_ylabel("Task index")
        ax.set_xticks(range(num_tasks))
        ax.set_yticks(range(num_tasks))

        plt.savefig("data/gradient_similarities.png", dpi=300)

        # Log the heatmap
        heatmap_logger = wandb.init(**self.wandb_params, name="gradient-alignment")
        wandb.log({"metrics-gradients/heatmap": wandb.Image("data/gradient_similarities.png")})
        heatmap_logger.finish()

        # Track how similarities evolved over time
        similarities_evolution = []
        for i in range(self.gradients.shape[0]):
            similarities = cosine_similarity(self.gradients[i, :, :])
            similarities_evolution.append(similarities)

        # Detailed logging per task
        for i in range(num_tasks):
            taskwise_training_logger = wandb.init(**self.wandb_params, name=f"task-{i}")
            global_step_name = "metrics-gradients/measurement-step"
            wandb.define_metric(global_step_name)

            wandb.log(
                {
                    f"metrics-gradients/initial_magnitude_normalized": self.initial_gradient_magnitudes_normalized[i],
                },
            )
            wandb.log(
                {
                    f"metrics-gradients/initial_magnitude": self.initial_gradient_magnitudes[i],
                },
            )

            for j in range(num_tasks):
                metric_name = f"metrics-gradients/task-{j}"
                wandb.define_metric(metric_name, step_metric=global_step_name)

            for k in range(self.gradients.shape[0]):  # Outer loop for the global step
                for j in range(num_tasks):
                    if i != j:
                        metric_name = f"metrics-gradients/task-{j}"
                        log_data = {
                            metric_name: similarities_evolution[k][i, j],
                            global_step_name: k,
                        }
                        wandb.log(log_data)

            taskwise_training_logger.finish()

    def produce_result(self):
        """
        Return all the angles we collected for post-processing or printing.
        """
        return self.results
