import torch
from common.datasets import RandomClassTasksCifar, TwentyImageTasksCifar, TenImageTasksCifar, FiveImageTasksCifar, get_cifar, FiftyImageTasksCifar, OneImageTasksCifar, TwoImageTasksCifar, get_SVHN
from common.metrics.florian_probing import FlorianProbing
from common.metrics.logging import BeautifulLogging, SweepyLogging
from common.metrics.gradients import GradientAlignment
from common.models import CIFAR_CNN_1
from common.train import CLTraining
import wandb
from common.metrics.layer_difference import LayerL2, LayerCosineDistance
import argparse
from common.replay_methods import SimpleReplay, LimitedReplay
from torch.utils.data import DataLoader, ConcatDataset, random_split

def main():

    parser = argparse.ArgumentParser()

    # Wandb parameters
    parser.add_argument("--wandb_entity", type=str, help="Wandb entity name", default="continual-learning-2024")
    parser.add_argument("--wandb_project", type=str, help="Wandb project name")
    parser.add_argument("--id", type=str, default="", help="ID for this wandb run (in case you want to run multiple runs with the same parameters)")
    parser.add_argument("--wandb_api_key", type=str, default="", help="Wandb api key for logging in")

    # Dataset and training parameters
    parser.add_argument("--dataset_size", type=int, choices=[1, 2, 5, 10, 20, 50], help="Number of classes per task")
    parser.add_argument("--cl_epochs", type=int, help="Number of epochs to train for all tasks during initial sequential training")
    parser.add_argument("--probing_epochs", type=int, help="Number of epochs to train on the probing dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")

    # Replay parameters
    parser.add_argument("--use_replay", action="store_true")
    parser.add_argument("--replay_buffer_size", type=int, help="Number of samples per class to store in the replay buffer")
    parser.add_argument("--replay_batch_size", type=int, help="How many samples to add from every batch from the replay buffer")
    parser.add_argument("--limited_replay", action="store_true", help="Use limited replay: remove the oldest task from buffer, if max capacity is reached")
    parser.add_argument("--task_limit", type=int, help="Maximum number of tasks to keep in the replay buffer, in case limited replay is used")

    # Extra data injection parameters
    parser.add_argument("--extra_data", action="store_true", help="Whether to add inject extra data to the tasks")
    parser.add_argument("--percent_new", type=float, help="The percentage of new data to include (range 0-1.0")
    parser.add_argument("--only_add_first_task", type=bool, default=False, help="Whether to add extra data to only task 1")

    # Metric parameters
    parser.add_argument("--use_gradient_alignment", action="store_true", help="Whether to use the gradient alignment metric")

    args = parser.parse_args()

    device = args.device

    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)


    # Here, define the common wandb parameters that will be called on wandb.init() in the CLTraining and
    # possibly other metrics.
    wandb_params = {
        # "project": "reproducing-florians-experiment-images5-fixed-4",
        "project": args.wandb_project,
        "entity": args.wandb_entity,
    }

    # Turn off wandb logging
    wandb.Settings(quiet=True)

    # Get the raw dataset
    full_train_dataset_cifar100, full_test_dataset_cifar100 = get_cifar('100')

    # Get the tasks dataset. Look at how these are implemented and just add new classes if you need to.
    if args.use_gradient_alignment:
        tasks = RandomClassTasksCifar(full_train_dataset_cifar100, full_test_dataset_cifar100, task_size=args.dataset_size)
    elif args.dataset_size == 5:
        tasks = FiveImageTasksCifar(full_train_dataset_cifar100, full_test_dataset_cifar100)
    elif args.dataset_size == 20:
        tasks = TwentyImageTasksCifar(full_train_dataset_cifar100, full_test_dataset_cifar100)
    elif args.dataset_size == 10:
        tasks = TenImageTasksCifar(full_train_dataset_cifar100, full_test_dataset_cifar100)
    elif args.dataset_size == 50:
        tasks = FiftyImageTasksCifar(full_train_dataset_cifar100, full_test_dataset_cifar100)
    elif args.dataset_size == 2:
        tasks = TwoImageTasksCifar(full_train_dataset_cifar100, full_test_dataset_cifar100)
    elif args.dataset_size == 1:
        tasks = OneImageTasksCifar(full_train_dataset_cifar100, full_test_dataset_cifar100)
    
    # If we need to, we have to add extra data/class to the tasks
    if args.extra_data:
        # Create the probing dataloader, as we will be now changing the tasks
        probing_dataloader_train = DataLoader(torch.utils.data.ConcatDataset([task[0].dataset for task in tasks.get_tasks_zipped(batch_size=args.batch_size)[:-1]]), batch_size=args.batch_size, shuffle=True)
        probing_dataloader_test = DataLoader(torch.utils.data.ConcatDataset([task[1].dataset for task in tasks.get_tasks_zipped(batch_size=args.batch_size)[:-1]]), batch_size=args.batch_size, shuffle=True)
        probing_dataloaders = (probing_dataloader_train, probing_dataloader_test)

        task_size, percent_new = args.dataset_size, args.percent_new
        
        if percent_new > 0:
            full_train_dataset_SVHN, full_test_dataset_SVHN = get_SVHN(set_one_class=True, new_label=100)
            split_SVHM_train = random_split(full_train_dataset_SVHN, [task_size / 100 for _ in range(100 // task_size)])
            split_SVHM_test = random_split(full_test_dataset_SVHN, [task_size / 100 for _ in range(100 // task_size)])
            
            # Combine SVHN dataset into task
            for i in range(len(tasks.train_datasets)):
                if args.only_add_first_task and i > 0:
                    break
                tasks.train_datasets[i] = ConcatDataset((tasks.train_datasets[i], random_split(split_SVHM_train[i], [percent_new, 1-percent_new])[0]))
                tasks.test_datasets[i] = ConcatDataset((tasks.test_datasets[i], random_split(split_SVHM_test[i], [percent_new, 1-percent_new])[0]))
                tasks.names[i] += ',house numbers'
        model_output_size = 101
    else:
        probing_dataloaders = None # Will be computed inside of FlorianProbing
        model_output_size = 100


    # Get the model. IMPORTANT: move it to the needed device HERE.
    # Do NOT edit the training loops to move the model to the device there, because
    # that would mess up FlorianProbing metric (as it needs references to layers).
    model = CIFAR_CNN_1(num_classes=model_output_size).to(device)

    # Define the Florian probing metric
    florian_probing_metric = FlorianProbing(
        # The order of layers to freeze. On the ith iteration, the first i layers are frozen.
        layers_order = [
            ([model.conv1], f"{args.id}-conv1"),
            ([model.conv2], f"{args.id}-conv2"),
            ([model.conv3], f"{args.id}-conv3"),
            ([model.fc1], f"{args.id}-fc1")
        ],

        # The list of all layers with parameters. Needed for reinitializtion of layers. Hurts to specify this, but I don't know a better way.
        all_layers = [model.conv1, model.conv2, model.conv3, model.fc1, model.fc2],

        # Define the parameters that will be needed when training on the FULL dataset (which comprises of all tasks in on dataset)
        optimizer=(torch.optim.Adam, {"lr": args.lr}),
        criterion=torch.nn.CrossEntropyLoss(),
        epochs=args.probing_epochs,
        batch_size=args.batch_size,

        device=device,
        wandb_params=wandb_params,

        # Will give nice visuals in W&B. Use True if you want to run sweeps, then it run all probing as a single run.
        sweepy_logging=False,
        custom_probing_dataloaders=probing_dataloaders
    )

    # Define the layer difference metrics
    layerL2 = LayerL2(
        layers=[
            (model.conv1, "conv1"),
            (model.conv2, "conv2"),
            (model.conv3, "conv3"),
            (model.fc1, "fc1"),
            (model.fc2, "fc2")
        ],
        wandb_params=wandb_params
    )

    layerCosine = LayerCosineDistance(
        layers=[
            (model.conv1, "conv1"),
            (model.conv2, "conv2"),
            (model.conv3, "conv3"),
            (model.fc1, "fc1"),
            (model.fc2, "fc2")
        ],
        wandb_params=wandb_params
    )

    # Define the replay buffer (if used)
    if args.use_replay:
        if args.limited_replay:
            replay = LimitedReplay(
                samples_per_class=args.replay_buffer_size,
                batch_size=args.replay_batch_size,
                max_dataloaders=args.task_limit
            )
        else:
            replay = SimpleReplay(
                samples_per_class=args.replay_buffer_size,
                batch_size=args.replay_batch_size
            )
    else:
        replay = None

    if args.use_gradient_alignment:
        gradients_alignment_metric = GradientAlignment(
            layers_order=[
                ([model.conv1], "conv1"),
                ([model.conv2], "conv2"),
                ([model.conv3], "conv3"),
                ([model.fc1], "fc1"),
            ],
            task_train_loaders=tasks.get_train_loaders(batch_size=args.batch_size, shuffle=True),
            criterion=torch.nn.CrossEntropyLoss(),
            device=device,
            check_every=50,  # e.g., every 1 batch
            wandb_params=wandb_params,
        )
        optional_metrics = [gradients_alignment_metric]
    else:
        optional_metrics = []

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
        epochs=args.cl_epochs,
        device=device,

        # Define the optimizer. It is done like this, so that I can reinitialize the optimizer for every task.
        # NOTE: this might not be optimal -> >>!
        optimizer=(torch.optim.Adam, {"lr": args.lr, "weight_decay": 0.001}),

        # Define the tasks to be trained on. This is a List[Pair[DataLoader, DataLoader]], with train and test loaders
        # for every task. Every "*TasksCifar" object will implement this get_tasks_zipped method.
        tasks=tasks.get_tasks_zipped(batch_size=args.batch_size),

        # Define the metrics to be used. For now, only FlorianProbing is implemented.
        metrics=[layerL2, layerCosine, florian_probing_metric, beautiful_logging] + optional_metrics,
        wandb_params=wandb_params,
        replay=replay
    )

    # Run the training
    results = cl_training.run()

    print("Metric results:", results[1])

if __name__ == "__main__":
    main()