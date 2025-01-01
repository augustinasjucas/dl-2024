from typing import List, Tuple
from torch.utils.data import DataLoader

class Metric:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def before_all_tasks(self, model, tasks: List[Tuple[DataLoader, DataLoader]], cl_training_object):
        # cl_training_object is the object that is calling this method - allows to take whatever you need from it
        pass

    def before_task(self, model, task_num, train_loader, test_loader):
        pass

    def after_batch(self, model, task_num, epoch_num, batch_num, batch_x, batch_y, batch_pred):
        pass

    def before_epoch(self, model, task_num, epoch_num):
        pass

    def after_epoch(self, model, task_num, epoch_num):
        pass

    def after_task(self, model, task_num, train_loader, test_loader):
        pass

    def after_all_tasks(self, model, tasks: List[Tuple[DataLoader, DataLoader]]):
        # do wandb logging here! you can create a new wandb run in here (or in produce_results)
        # If you do wandb logging in previous methods, you might get problems!
        pass

    def get_name(self):
        return self.name

    def produce_result(self):
        assert 0 == 1, "Produce result not implemented!"
