import argparse

import torch

import wandb
from common.datasets import get_experiment_setup
from common.metrics.florian_probing import FlorianProbing
from common.metrics.logging import BeautifulLogging
from common.models import SimpleCNN
from common.train import CLTraining


def train_experiment(experiment_number=1):
    # Check if this is being run as part of a sweep
    is_sweep = wandb.run is not None

    # Base wandb parameters
    wandb_params = {
        "project": "size-variations",
        "entity": "continual-learning-2024",
    }

    if is_sweep:
        # We're in a sweep, get the experiment number from sweep config
        run = wandb.run
        experiment_number = run.config.experiment_number
        # Add group to wandb params for organized logging
        wandb_params["group"] = f"sweep-{run.id}"
    else:
        # Normal run, initialize wandb
        wandb.init(**wandb_params)

    experiment_number = 1
    # Get the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the raw dataset and tasks for the experiment
    tasks = get_experiment_setup(
        "cifar100", batch_size=256, experiment_number=experiment_number
    )

    # Get the model and move it to device
    model = SimpleCNN().to(device)

    # Define the Florian probing metric with beautiful logging
    florian_probing_metric = FlorianProbing(
        layers_order=[
            ([model.conv1], "conv1"),
            ([model.conv2], "conv2"),
            ([model.conv3], "conv3"),
            ([model.fc1], "fc1"),
        ],
        all_layers=[
            model.conv1,
            model.conv2,
            model.conv3,
            model.fc1,
            model.fc2,
        ],
        optimizer=(torch.optim.Adam, {"lr": 0.001}),
        criterion=torch.nn.CrossEntropyLoss(),
        epochs=100,
        batch_size=256,
        device=device,
        wandb_params=wandb_params,
        sweepy_logging=False,  # Always use beautiful logging
    )

    # Use BeautifulLogging for task-wise logging
    beautiful_logging = BeautifulLogging(wandb_params)

    # Define the CL training task
    cl_training = CLTraining(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        epochs=100,
        device=device,
        optimizer=(torch.optim.Adam, {"lr": 0.001, "weight_decay": 0.001}),
        tasks=tasks,
        metrics=[florian_probing_metric, beautiful_logging],
        wandb_params=wandb_params,
    )

    # Run the training
    results = cl_training.run()

    # Log the overall results if in sweep mode
    if is_sweep:
        wandb.log({"experiment_number": experiment_number, "final_results": results[1]})

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", type=int, default=1, help="Experiment number for normal runs"
    )
    args = parser.parse_args()

    train_experiment(experiment_number=args.experiment)
