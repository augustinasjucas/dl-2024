import wandb
from common.metrics import Metric
from common.utils import simple_test
from torch.utils.data import DataLoader

class LossAccuracyCalculator(Metric):
    def before_all_tasks(self, model, tasks, cl_training_object):
        self.criterion = cl_training_object.criterion
        self.device = cl_training_object.device

    def before_epoch(self, model, task_num, epoch_num):
        self.epoch_loss = 0
        self.epoch_accuracy = 0

    def after_batch(self, model, task_num, epoch_num, batch_num, batch_x, batch_y, batch_pred):
        loss = self.criterion(batch_pred, batch_y)
        self.epoch_loss += loss.item()
        pred = batch_pred.argmax(dim=1, keepdim=True)
        self.epoch_accuracy += pred.eq(batch_y.view_as(pred)).sum().item()

    def produce_result(self):
        return "-"

class BeautifulLogging(LossAccuracyCalculator):
    def __init__(self, wandb_params):
        super().__init__("BeautifulLogging", "Logs the training progress to wandb, with each task being a separate RUN!")
        self.wandb_params = wandb_params

    def before_task(self, model, task_index, train_loader, test_loader):
        self.current_task_logger = wandb.init(
                **self.wandb_params,
                name=f"sequential-training-task-{task_index}"
            )
        self.train_loader = train_loader

    def after_task(self, model, task_index, train_loader, test_loader):
         # Test on this task
        test_loss, test_accuracy = simple_test(model, test_loader, self.criterion, self.device)
        train_loss, train_accuracy = simple_test(model, train_loader, self.criterion, self.device)

        wandb.log({
            f"primary-sequential-training/test-loss": test_loss,
            f"primary-sequential-training/test-accuracy": test_accuracy,
            f"primary-sequential-training/train-loss": train_loss,
            f"primary-sequential-training/train-accuracy": train_accuracy
        })

        # Print the results for this task
        print(f"    After training on task {task_index}: Test Loss: {test_loss}. Test Accuracy: {test_accuracy}, Train Loss: {train_loss}. Train Accuracy: {train_accuracy}")

        self.current_task_logger.finish()

    def after_epoch(self, model, task_num, epoch_num):
        # Log the epoch results to wandb
        wandb.log({
            f"primary-sequential-training/loss": self.epoch_loss / len(self.train_loader.dataset),
            f"primary-sequential-training/accuracy": self.epoch_accuracy / len(self.train_loader.dataset),
        }, step=epoch_num)


class SweepyLogging(LossAccuracyCalculator):
    def __init__(self, wandb_params):
        super().__init__("BeautifulLogging", "Logs the training progress to wandb, with each task being a separate RUN!")
        self.wandb_params = wandb_params

    def before_all_tasks(self, model, tasks, cl_training_object):
        super().before_all_tasks(model, tasks, cl_training_object)
        wandb.init(**self.wandb_params)

    def before_task(self, model, task_index, train_loader, test_loader):
        self.task_index = task_index
        self.train_loader = train_loader

    def after_task(self, model, task_index, train_loader, test_loader):
         # Test on this task
        test_loss, test_accuracy = simple_test(model, test_loader, self.criterion, self.device)
        train_loss, train_accuracy = simple_test(model, train_loader, self.criterion, self.device)

        wandb.log({
            f"primary-sequential-training/task{task_index}/test-loss": test_loss,
            f"primary-sequential-training/task{task_index}/test-accuracy": test_accuracy,
            f"primary-sequential-training/task{task_index}/train-loss": train_loss,
            f"primary-sequential-training/task{task_index}/train-accuracy": train_accuracy
        })

        # Print the results for this task
        print(f"    After training on task {task_index}: Test Loss: {test_loss}. Test Accuracy: {test_accuracy}, Train Loss: {train_loss}. Train Accuracy: {train_accuracy}")

    def after_epoch(self, model, task_num, epoch_num):
        # Log the epoch results to wandb
        wandb.log({
            f"primary-sequential-training/task{self.task_index}/loss": self.epoch_loss / len(self.train_loader.dataset),
            f"primary-sequential-training/task{self.task_index}/accuracy": self.epoch_accuracy / len(self.train_loader.dataset),
        })
