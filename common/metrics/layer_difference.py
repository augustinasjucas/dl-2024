from common.metrics import Metric
import copy
import wandb
import torch
from math import sqrt

class LayerL2(Metric):
    # As the sequential training progresses, calculates for every layer the average difference between
    # the weights before and after training on a task. Ignores the difference for the first task as the model
    # is random before it.

    def __init__(self, layers, wandb_params):
        super().__init__("LayerL2", "Calculates the L2 norm of the difference between the weights of a layer before and after training on a task. " +
                         "Does that for every task while training sequentially and calculates the average for every layer. Returns that")
        # assert that none of the layer names are the same
        layer_names = [name for _, name in layers]
        assert len(layer_names) == len(set(layer_names)), "Layer names have to be unique"

        self.layers = layers
        self.results = {}
        self.task_count = 0
        self.saved_layers = {}
        self.wandb_params = wandb_params

    def before_task(self, model, task_num, train_loader, test_loader):
        self.task_count += 1
        self.saved_layers = {}
        if self.task_count == 1:
            return

        for layer, name in self.layers:
            self.saved_layers[name] = copy.deepcopy(list(layer.parameters()))

    def after_task(self, model, task_num, train_loader, test_loader):
        if self.task_count == 1:
            return

        for layer, name in self.layers:
            # note that layer.parameters() returns a generator, so we have to convert
            # it to a list to be able to subtract the saved parameters
            # print out the compared parameters
            diff = torch.cat([p.flatten() - saved_p.flatten() for p, saved_p in zip(layer.parameters(), self.saved_layers[name])])
            num_params = sum([p.numel() for p in layer.parameters()])
            l2 = diff.norm(2).item() / sqrt(num_params)
            if name not in self.results:
                self.results[name] = []
            self.results[name].append(l2)

    def after_all_tasks(self, model, tasks):
        for layer, name in self.layers:
            wandb.init(
                **self.wandb_params,
                name="layer-" + name
            )

            # Log all differences for this layer between tasks
            for i, l2 in enumerate(self.results[name]):
                wandb.log({
                    f"metrics-layer_difference_L2/differences-between-tasks": l2
                })

            # Log only the average to get nice plots
            wandb.log({
                f"metrics-layer_difference_L2/average-difference": sum(self.results[name]) / len(self.results[name])
            })

            wandb.finish()

    def produce_result(self):
        return self.results


class LayerCosineDistance(Metric):
    # As the sequential training progresses, calculates for every layer the average difference between
    # the weights before and after training on a task. Ignores the difference for the first task as the model
    # is random before it.

    def __init__(self, layers, wandb_params):
        super().__init__("LayerL2", "Calculates the cosine distance of the difference between the weights of a layer before and after training on a task. " +
                         "Does that for every task while training sequentially and calculates the average for every layer. Returns that")
        # assert that none of the layer names are the same
        layer_names = [name for _, name in layers]
        assert len(layer_names) == len(set(layer_names)), "Layer names have to be unique"

        self.layers = layers
        self.results = {}
        self.task_count = 0
        self.saved_layers = {}
        self.wandb_params = wandb_params

    def before_task(self, model, task_num, train_loader, test_loader):
        self.task_count += 1
        self.saved_layers = {}
        for layer, name in self.layers:
            self.saved_layers[name] = copy.deepcopy(list(layer.parameters()))

    def after_task(self, model, task_num, train_loader, test_loader):


        for layer, name in self.layers:
            # note that layer.parameters() returns a generator, so we have to convert
            # it to a list to be able to subtract the saved parameters
            # print out the compared parameters

            vec1 = torch.cat([p.flatten() for p in layer.parameters()])
            vec2 = torch.cat([p.flatten() for p in self.saved_layers[name]])
            diff = torch.dot(vec1, vec2) / (vec1.norm(2) * vec2.norm(2))
            if name not in self.results:
                self.results[name] = []
            self.results[name].append(diff)

    def after_all_tasks(self, model, tasks):
        for layer, name in self.layers:
            wandb.init(
                **self.wandb_params,
                name="layer-" + name
            )

            # Log all differences for this layer between tasks
            for i, l2 in enumerate(self.results[name]):
                wandb.log({
                    f"metrics-layer_difference_cosine_similarity/differences-between-tasks": l2
                })

            # Log only the average to get nice plots
            wandb.log({
                f"metrics-layer_difference_cosine_similarity/average-difference": sum(self.results[name]) / len(self.results[name])
            })

            wandb.finish()

    def produce_result(self):
        return self.results

# How to use this metric:
# layerL2 = LayerL2(
#     layers=[
#         (model.conv1, "conv1"),
#         (model.conv2, "conv2"),
#         (model.conv3, "conv3"),
#         (model.fc1, "fc1"),
#         (model.fc2, "fc2")
#     ],
#     wandb_params=wandb_params
# )
# it will just produce some more data to wandb : )
