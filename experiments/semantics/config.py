
import argparse

def get_config():
    parser = argparse.ArgumentParser()
    # Sweep
    parser.add_argument("--dataset", type=str, default='cifar100')
    parser.add_argument("--starting_superclass", type=int, default=18)  # what superclass to use first (only supporting CIFAR 100 currently)
    parser.add_argument("--model_name", type=str, default='ResNet')
    
    # Model training
    parser.add_argument("--multihead", type=bool, default=True, help='Whether a different head is used for each task')
    parser.add_argument("--optimizer_name", type=str, default='Adam', help="Optimizer for training")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    # parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate")

    return vars(parser.parse_args())