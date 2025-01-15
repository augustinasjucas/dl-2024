import torch
import wandb
from common.metrics import Metric
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from typing import List, Tuple
import os


class GradientAlignment(Metric):
    def __init__(
        self,
        layers_order: List[Tuple[List[torch.nn.Module], str]],  # List of (layers, name) tuples
        task_train_loaders,  # List of DataLoader, one for each task
        criterion,  # e.g. torch.nn.CrossEntropyLoss()
        device="cuda",
        check_every=100,  # how often to check (e.g. every 100 batches)
        wandb_params: dict = None,
    ):
        """
        A metric that computes pairwise angles between gradients from each task.
        Additionally, it can measure the gradient magnitude at the start of each task
        to get a sense of transfer (possibly normalized by parameter norm).

        Instead of checking every n epochs, we check every n batches.
        """
        super().__init__("Gradient Alignment", "")
        self.task_train_loaders = task_train_loaders
        self.criterion = criterion
        self.device = device
        self.check_every = check_every
        self.wandb_params = wandb_params

        self.gradients = None
        self.results = None

        # For storing per-layer gradients over time: {layer_name: np.ndarray of shape [time, tasks, layer_params]}
        self.layerwise_gradients = {layer_name: None for (_, layer_name) in layers_order}
        self.layers_order = layers_order

        # Store gradient magnitudes at the beginning of each task.
        self.initial_gradient_magnitudes_normalized = []
        self.initial_gradient_magnitudes = []

        # Store the norms of the input gradients
        self.input_grad_norms = []

        # A global counter for batches (across tasks)
        self.global_batch_count = 0

        # Utilities for random batch sampling
        self.current_epoch_iterators = {}

    def _get_random_batch(self, task_num):
        x, y = next(self.current_epoch_iterators[task_num])
        return x.to(self.device), y.to(self.device)

    def _compute_gradient(self, model, x, y):
        """
        Compute the gradient for the entire model, flatten all parameter grads
        into a single 1D tensor. Return that flattened gradient.
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

    def _compute_gradients_for_all_layers(self, model, x, y):
        """
        Perform a forward/backward pass once, then collect:
          1) Flattened gradient for the entire model
          2) Flattened gradient for each set of layers in `self.layers_order`
        Return (global_gradient, {layer_name: layer_gradient}).
        """
        # Zero out existing gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Forward + backward
        outputs = model(x)
        loss = self.criterion(outputs, y)
        loss.backward()

        # Entire model gradient
        global_grads = []
        for p in model.parameters():
            if p.grad is not None:
                global_grads.append(p.grad.view(-1))
        global_grads = torch.cat(global_grads, dim=0)

        # Layerwise gradients
        layerwise_grads_dict = {}
        for layers, layer_name in self.layers_order:
            layer_params_grads = []
            for layer in layers:
                for p in layer.parameters():
                    if p.grad is not None:
                        layer_params_grads.append(p.grad.view(-1))
            if len(layer_params_grads) > 0:
                layerwise_grads_dict[layer_name] = torch.cat(layer_params_grads, dim=0)
            else:
                # If for some reason the layer has no grads, store a zero vector
                layerwise_grads_dict[layer_name] = torch.zeros(1, device=self.device)

        return global_grads, layerwise_grads_dict

    def _compute_input_gradient_norm(self, model, x, y):
        """
        Compute the L2-norm of dLoss/dX. Return the average norm over the batch.
        """
        # Ensure the input has grad tracking
        x.requires_grad_(True)

        # Zero model grads
        model.zero_grad()

        # Forward & compute loss
        out = model(x)
        loss = self.criterion(out, y)

        # Backward to get dLoss/dX
        loss.backward()

        # Compute per-example norms, then average
        grad_norms = x.grad.view(x.size(0), -1).norm(dim=1, p=2)
        avg_grad_norm = grad_norms.mean().item()

        # Detach from autograd, turn off requires_grad
        x.requires_grad_(False)

        return avg_grad_norm

    def after_batch(self, model, curr_task_num, epoch_idx, batch_idx, x, y, y_pred):
        self.global_batch_count += 1

        if self.global_batch_count % self.check_every != 0:
            return

        current_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Prepare lists that will hold gradients for each task (some will be None for the current task)
        gradients_list = [None] * len(self.task_train_loaders)
        layerwise_gradients_dict_of_lists = {
            layer_name: [None] * len(self.task_train_loaders) for (_, layer_name) in self.layers_order
        }

        # Compute gradients for tasks != curr_task_num
        for task_num in range(len(self.task_train_loaders)):
            if task_num == curr_task_num:
                # Skip measuring the current task
                continue

            batch_x, batch_y = self._get_random_batch(task_num)
            g_global, g_layers = self._compute_gradients_for_all_layers(model, batch_x, batch_y)

            gradients_list[task_num] = g_global.detach().cpu().numpy()
            for layer_name in g_layers:
                layerwise_gradients_dict_of_lists[layer_name][task_num] = g_layers[layer_name].detach().cpu().numpy()

            # Restore model weights
            model.load_state_dict(current_state)

        # Convert to numpy array, filling None entries (current task) with zeros
        valid_global_grad = next((g for g in gradients_list if g is not None), None)
        if valid_global_grad is None:
            return  # No valid gradient at all (unlikely, unless there's only one task)

        for i, g in enumerate(gradients_list):
            if g is None:
                gradients_list[i] = np.zeros_like(valid_global_grad)  # fill skipped task slot with zeros

        new_global_gradients = np.stack(gradients_list, axis=0)  # shape: [tasks, total_params]
        new_global_gradients = np.expand_dims(new_global_gradients, axis=0)  # [1, tasks, total_params]

        if self.gradients is not None:
            self.gradients = np.concatenate((self.gradients, new_global_gradients), axis=0)
        else:
            self.gradients = new_global_gradients

        # Handle each layer similarly, filling None with zeros
        for _, layer_name in self.layers_order:
            layer_list = layerwise_gradients_dict_of_lists[layer_name]
            valid_layer_grad = next((g for g in layer_list if g is not None), None)
            if valid_layer_grad is None:
                continue
            for i, g in enumerate(layer_list):
                if g is None:
                    layer_list[i] = np.zeros_like(valid_layer_grad)
            layerwise_array = np.stack(layer_list, axis=0)  # [tasks, layer_params]
            layerwise_array = np.expand_dims(layerwise_array, axis=0)  # [1, tasks, layer_params]

            if self.layerwise_gradients[layer_name] is not None:
                self.layerwise_gradients[layer_name] = np.concatenate(
                    (self.layerwise_gradients[layer_name], layerwise_array), axis=0
                )
            else:
                self.layerwise_gradients[layer_name] = layerwise_array

        model.load_state_dict(current_state)

    def before_task(self, model, task_num, train_loader, test_loader):
        """
        Measure the gradient magnitude on this new task (using a random batch),
        normalized by parameter norm.
        """
        # Save the current model state
        current_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Get a random batch for this task
        x, y = next(iter(train_loader))
        x, y = x.to(self.device), y.to(self.device)

        g = self._compute_gradient(model, x, y)

        # Compute norms
        with torch.no_grad():
            param_norm = 0.0
            for p in model.parameters():
                param_norm += p.data.norm() ** 2
            param_norm = param_norm.sqrt()

            grad_norm = g.norm()
            grad_norm_normalized = grad_norm / (param_norm + 1e-8)

        self.initial_gradient_magnitudes_normalized.append(grad_norm_normalized.item())
        self.initial_gradient_magnitudes.append(grad_norm.item())

        # Restore model
        model.load_state_dict(current_state)

        # Measure input gradient norm
        input_grad_norm = self._compute_input_gradient_norm(model, x, y)
        self.input_grad_norms.append(input_grad_norm)

        # Restore model
        model.load_state_dict(current_state)

    def before_epoch(self, model, task_num, epoch_num):
        for i, loader in enumerate(self.task_train_loaders):
            self.current_epoch_iterators[i] = iter(loader)

    def after_task(self, model, task_num, train_loader, test_loader):
        pass

    def after_all_tasks(self, model, tasks):
        """
        Called once after all tasks are completed. We compute the average
        pairwise cosine similarities for each task pair (global), and also
        for each layer. Then we log them all to W&B, producing separate
        heatmaps. All color scales are set from -1 to 1 for easy comparison.
        """
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)

        num_tasks = len(self.task_train_loaders)

        min_sim = 1
        max_sim = -1

        # Compute average pairwise similarities for the global gradients
        global_similarities = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(num_tasks):
                # Collect all timesteps where task i has non-zero grads
                i_indices = [t for t in range(self.gradients.shape[0]) if not np.allclose(self.gradients[t, i], 0)]
                # Collect all timesteps where task j has non-zero grads
                j_indices = [t for t in range(self.gradients.shape[0]) if not np.allclose(self.gradients[t, j], 0)]

                # ============ Diagonal case (i == i) ============
                if i == j:
                    # If there's fewer than 2 time steps, we can't form pairs
                    if len(i_indices) < 2:
                        global_similarities[i, i] = 0.0
                    else:
                        # Compare each pair of (i_indices) among themselves
                        grads_i = self.gradients[i_indices, i, :]  # shape: [N, D]
                        sims_ii = cosine_similarity(grads_i, grads_i)  # [N, N]

                        # Optionally exclude each vector's self-sim on the diagonal
                        mask = np.ones_like(sims_ii, dtype=bool)
                        np.fill_diagonal(mask, 0)
                        if np.any(mask):
                            global_similarities[i, i] = np.mean(sims_ii[mask])
                        else:
                            global_similarities[i, i] = 0.0

                # ============ Off-diagonal (i != j) ============
                else:
                    # If either i_indices or j_indices is empty => no data => sim = 0
                    if len(i_indices) == 0 or len(j_indices) == 0:
                        global_similarities[i, j] = 0.0
                    else:
                        grads_i = self.gradients[i_indices, i, :]
                        grads_j = self.gradients[j_indices, j, :]
                        sims_ij = cosine_similarity(grads_i, grads_j)  # [len(i_indices), len(j_indices)]
                        global_similarities[i, j] = np.mean(sims_ij)

        # 2) Layerwise similarity calculation:
        layerwise_similarities = np.zeros((len(self.layers_order), num_tasks, num_tasks))
        for idx, (layer_list, layer_name) in enumerate(self.layers_order):
            layer_array = self.layerwise_gradients[layer_name]  # shape: [time, tasks, layer_params]

            for i in range(num_tasks):
                for j in range(num_tasks):
                    i_indices = [t for t in range(layer_array.shape[0]) if not np.allclose(layer_array[t, i], 0)]
                    j_indices = [t for t in range(layer_array.shape[0]) if not np.allclose(layer_array[t, j], 0)]

                    if i == j:
                        # Intra-task (compare each time step for task i to itself)
                        if len(i_indices) < 2:
                            layerwise_similarities[idx, i, i] = 0.0
                        else:
                            grads_i = layer_array[i_indices, i, :]
                            sims_ii = cosine_similarity(grads_i, grads_i)
                            mask = np.ones_like(sims_ii, dtype=bool)
                            np.fill_diagonal(mask, 0)
                            if np.any(mask):
                                layerwise_similarities[idx, i, i] = np.mean(sims_ii[mask])
                            else:
                                layerwise_similarities[idx, i, i] = 0.0
                    else:
                        # Off-diagonal
                        if len(i_indices) == 0 or len(j_indices) == 0:
                            layerwise_similarities[idx, i, j] = 0.0
                        else:
                            grads_i = layer_array[i_indices, i, :]
                            grads_j = layer_array[j_indices, j, :]
                            sims_ij = cosine_similarity(grads_i, grads_j)
                            layerwise_similarities[idx, i, j] = np.mean(sims_ij)

        min_sim = min(min_sim, np.min(global_similarities), np.min(layerwise_similarities))
        max_sim = max(max_sim, np.max(global_similarities), np.max(layerwise_similarities))

        # Plot the global heatmap
        fig_global, ax_global = plt.subplots(figsize=(8, 6), dpi=300)
        cax_global = ax_global.matshow(
            global_similarities, cmap="viridis", interpolation="none", vmin=min_sim, vmax=max_sim
        )
        fig_global.colorbar(cax_global)
        ax_global.set_title("Average cosine similarity (global) between tasks")
        ax_global.set_xlabel("Task index")
        ax_global.set_ylabel("Task index")
        ax_global.set_xticks(range(num_tasks))
        ax_global.set_yticks(range(num_tasks))
        plt.savefig(f"{output_dir}/global_similarities.png", dpi=300)

        # Log the global heatmap
        global_logger = wandb.init(**self.wandb_params, name="gradient-alignment-global")
        wandb.log({"metrics-gradients/global_similarities": wandb.Image(f"{output_dir}/global_similarities.png")})
        artifact_global = wandb.Artifact("global_similarities", type="metric")
        np.save(f"{output_dir}/global_similarities.npy", global_similarities)
        artifact_global.add_file(f"{output_dir}/global_similarities.npy")
        global_logger.log_artifact(artifact_global)
        global_logger.finish()

        # Plot the per-layer heatmaps
        fig_layers, axes = plt.subplots(1, len(self.layers_order), figsize=(5 * len(self.layers_order), 5), dpi=300)
        if len(self.layers_order) == 1:
            axes = [axes]  # Make it iterable

        for idx, (layer_list, layer_name) in enumerate(self.layers_order):
            ax = axes[idx]
            cax_layer = ax.matshow(
                layerwise_similarities[idx], cmap="viridis", interpolation="none", vmin=min_sim, vmax=max_sim
            )
            ax.set_title(f"{layer_name}")
            ax.set_xlabel("Task index")
            ax.set_ylabel("Task index")
            ax.set_xticks(range(num_tasks))
            ax.set_yticks(range(num_tasks))
            fig_layers.colorbar(cax_layer, ax=ax)
        plt.savefig(f"{output_dir}/layerwise_similarities.png", dpi=300)

        # Log the per-layer heatmap
        layerwise_logger = wandb.init(**self.wandb_params, name="gradient-alignment-per-layer")
        wandb.log({"metrics-gradients/layerwise_similarities": wandb.Image(f"{output_dir}/layerwise_similarities.png")})
        artifact_layerwise = wandb.Artifact("layerwise_similarities", type="metric")
        np.save(f"{output_dir}/layerwise_similarities.npy", layerwise_similarities)
        artifact_layerwise.add_file(f"{output_dir}/layerwise_similarities.npy")
        layerwise_logger.log_artifact(artifact_layerwise)
        layerwise_logger.finish()

        # Track how similarities evolved over time (global)
        running_similarities = []
        for i in range(self.gradients.shape[0]):
            similarities = cosine_similarity(self.gradients[i, :, :])
            running_similarities.append(similarities)

        # Detailed logging per task
        for i in range(num_tasks):
            taskwise_training_logger = wandb.init(**self.wandb_params, name=f"task-{i}")
            global_step_name = "metrics-gradients/measurement-step"
            wandb.define_metric(global_step_name)

            wandb.log({"metrics-gradients/input_gradient_magnitude": self.input_grad_norms[i]})
            wandb.log(
                {
                    "metrics-gradients/initial_magnitude_normalized": self.initial_gradient_magnitudes_normalized[i],
                }
            )
            wandb.log(
                {
                    "metrics-gradients/initial_magnitude": self.initial_gradient_magnitudes[i],
                }
            )

            for j in range(num_tasks):
                metric_name = f"metrics-gradients/similarity_with_task-{j}"
                wandb.define_metric(metric_name, step_metric=global_step_name)

            # Now log how the similarity with other tasks evolves over time
            for k in range(self.gradients.shape[0]):  # Outer loop for the global step
                for j in range(num_tasks):
                    if i != j:
                        metric_name = f"metrics-gradients/similarity_with_task-{j}"
                        log_data = {
                            metric_name: running_similarities[k][i, j],
                            global_step_name: k,
                        }
                        wandb.log(log_data)

            taskwise_training_logger.finish()

        self.results = {
            "global_similarities": global_similarities,
            "layerwise_similarities": layerwise_similarities,
        }

    def produce_result(self):
        """
        Return all the angles (similarities) we collected for post-processing or printing.
        """
        return self.results
