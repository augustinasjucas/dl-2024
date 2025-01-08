import torch
from common.datasets import (
    FiftyImageTasksCifar,
    TwentyImageTasksCifar,
    TenImageTasksCifar,
    FiveImageTasksCifar,
    get_cifar,
)
from common.metrics.florian_probing import FlorianProbing
from common.metrics.gradients import GradientAlignment
from common.metrics.logging import BeautifulLogging, SweepyLogging
from common.models import CIFAR_CNN_1
from common.train import CLTraining
import wandb


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Here, define the common wandb parameters that will be called on wandb.init() in the CLTraining and
    # possibly other metrics.
    wandb_params = {
        "project": "gradients-test-94",
        "entity": "continual-learning-2024",
    }

    # Turn off wandb logging
    wandb.Settings(quiet=True)
    # wandb.init(mode="disabled")

    # Get the raw dataset
    full_train_dataset_cifar100, full_test_dataset_cifar100 = get_cifar("100")

    # Get the tasks dataset. Look at how these are implemented and just add new classes if you need to.
    batch_size = 256
    task = TwentyImageTasksCifar(full_train_dataset_cifar100, full_test_dataset_cifar100)
    tasks_zipped = task.get_tasks_zipped(batch_size=batch_size)
    train_loaders = task.get_train_loaders(batch_size=batch_size)

    # Get the model. IMPORTANT: move it to the needed device HERE.
    # Do NOT edit the training loops to move the model to the device there, because
    # that would mess up FlorianProbing metric (as it needs references to layers).
    model = CIFAR_CNN_1().to(device)

    # Define the Florian probing metric
    florian_probing_metric = FlorianProbing(
        # The order of layers to freeze. On the ith iteration, the first i layers are frozen.
        layers_order=[
            ([model.conv1], "conv1"),
            ([model.conv2], "conv2"),
            ([model.conv3], "conv3"),
            ([model.fc1], "fc1"),
        ],
        # The list of all layers with parameters. Needed for reinitializtion of layers. Hurts to specify this, but I don't know a better way.
        all_layers=[model.conv1, model.conv2, model.conv3, model.fc1, model.fc2],
        # Define the parameters that will be needed when training on the FULL dataset (which comprises of all tasks in on dataset)
        optimizer=(torch.optim.Adam, {"lr": 0.001}),
        criterion=torch.nn.CrossEntropyLoss(),
        epochs=80,
        batch_size=256,
        device=device,
        wandb_params=wandb_params,
        # Will give nice visuals in W&B. Use True if you want to run sweeps, then it run all probing as a single run.
        sweepy_logging=False,
    )

    gradients_alignment_metric = GradientAlignment(
        layers_order=[
            ([model.conv1], "conv1"),
            ([model.conv2], "conv2"),
            ([model.conv3], "conv3"),
            ([model.fc1], "fc1"),
        ],
        task_train_loaders=train_loaders,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        check_every=1,  # e.g., every 1 epoch
        wandb_params=wandb_params,
    )

    # Define the metric that will take care of monitoring accuracy as loss during sequential training
    # the optins here are: BeautifulLogging OR SweepyLogging. SweepyLogging will log the sequential training
    # as a single run (i.e., properly! just as it should be in W&B). Use Sweepy, when you need to do sweeps.
    # However, if you want nice graphs and visuals, use Beautiful logging. Similarly, in FlorianProbing, you can
    # set sweepy_logging=False, if you want nice visuals, but if you want to run sweeps, use sweep_logging=True
    beautiful_logging = BeautifulLogging(wandb_params)

    # Define the CL training task
    cl_training = CLTraining(
        # Give the model. IMPORTANT: has to be the same object as in FlorianProbing metric (not a copy!)
        model=model,
        # Define the parameters for training every task (same criterion and same # of epochs for all tasks)
        criterion=torch.nn.CrossEntropyLoss(),
        epochs=80,
        device=device,
        # Define the optimizer. It is done like this, so that I can reinitialize the optimizer for every task.
        # NOTE: this might not be optimal -> >>!
        optimizer=(torch.optim.Adam, {"lr": 0.001, "weight_decay": 0.001}),
        # Define the tasks to be trained on. This is a List[Pair[DataLoader, DataLoader]], with train and test loaders
        # for every task. Every "*TasksCifar" object will implement this get_tasks_zipped method.
        tasks=tasks_zipped,
        # Define the metrics to be used. For now, only FlorianProbing is implemented.
        metrics=[gradients_alignment_metric, beautiful_logging],
        wandb_params=wandb_params,
    )

    # Run the training
    results = cl_training.run()

    print("Metric results:", results[1])


if __name__ == "__main__":

    # For sweeps: (make sure to use SweepyLogging in the CLTraining object and to set sweepy_logging=True in FlorianProbing)
    # # Define the search space
    # sweep_configuration = {
    #     "method": "random",
    #     "metric": {"goal": "minimize", "name": "score"},
    #     "parameters": {
    #         "x": {"max": 0.1, "min": 0.01},
    #         "y": {"values": [1, 3, 7]},
    #     },
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="stupid-sweep")
    # wandb.agent(sweep_id, function=main, count=10)

    main()
