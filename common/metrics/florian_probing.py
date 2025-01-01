from common.metrics import Metric
import torch
from typing import List, Tuple
import copy
from torch.utils.data import DataLoader
from common.utils import simple_train, simple_test
import wandb

class FlorianProbing(Metric):
    def __init__(self,
                 layers_order: List[ Tuple[List[torch.nn.Module], str]],
                 all_layers: List[torch.nn.Module],
                 optimizer: Tuple[torch.optim.Optimizer, dict],
                 criterion: torch.nn.Module,
                 epochs: int,
                 batch_size: int = 256,
                 device: str = "cuda",
                 wandb_params: dict = None,
                 sweepy_logging: bool = False
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
            """
        )
        self.results = []

    def after_all_tasks(self, model, tasks: List[Tuple[DataLoader, DataLoader]]):

        # Construct a full train and test loader with all data from all tasks
        full_train_loader = DataLoader(torch.utils.data.ConcatDataset([task[0].dataset for task in tasks]), batch_size=self.batch_size, shuffle=True)
        full_test_loader = DataLoader(torch.utils.data.ConcatDataset([task[1].dataset for task in tasks]), batch_size=self.batch_size, shuffle=True)

        all_layers_to_freeze = []

        for new_layers_to_freeze, name in self.layers_order:

            if self.sweepy_logging:
                sweepy_string = f"layer-{name}/"
            else:
                layerwise_training_logger = wandb.init(**self.wandb_params, name=f"layer-{name}")
                sweepy_string = ""

            all_layers_to_freeze.extend(new_layers_to_freeze)

            # Make a copy of the model for later reconstruction of "model"
            model_copy = copy.deepcopy(model)

            # first, freeze all needed layers
            for layer in all_layers_to_freeze:
                layer.requires_grad = False
            # then reset the parameters of all layers that are not frozen
            for layer in self.all_layers:
                if layer not in all_layers_to_freeze:
                    layer.reset_parameters()

            # now train the model on all data from all tasks
            optimizer = self.optimizer[0](model.parameters(), **self.optimizer[1])
            simple_train(model, full_train_loader, optimizer, self.criterion, self.epochs, self.device, f"metrics-florian_probing/intermediate-training-results/{sweepy_string}")
            loss, acc  = simple_test(model, full_test_loader,  self.criterion, self.device)
            loss_train, acc_train = simple_test(model, full_train_loader,  self.criterion, self.device)

            # append the results
            self.results.append((name, acc, loss, acc_train, loss_train))

            wandb.log({
                f"metrics-florian_probing/final-results/{sweepy_string}test-accuracy": acc, f"metrics-florian_probing/final-results/{sweepy_string}test-loss": loss,
                f"metrics-florian_probing/final-results/{sweepy_string}train-accuracy": acc_train, f"metrics-florian_probing/final-results/{sweepy_string}train-loss": loss_train,
                f"metrics-florian_probing/final-results/{sweepy_string}layer-index": len(self.results) - 1
            })

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

    def after_batch(self, model, task_num, epoch_num, batch_num, batch_x, batch_y, batch_pred):
        pass

    def after_epoch(self, model, task_num, epoch_num):
        pass

    def before_task(self, model, task_num, train_loader, test_loader):
        pass

    def after_task(self, model, task_num, train_loader, test_loader):
        pass

