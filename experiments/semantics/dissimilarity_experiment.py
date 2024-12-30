
import random
import gc
import torch
import matplotlib.pyplot as plt
# import sklearn.metrics as metrics
import wandb

from torch import nn
from copy import deepcopy

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import get_config
from common.models import ResNet, SemanticCNN
from common.datasets import get_cifar
from common.dataloader import split_data, reshuffle

# setup config params (what is needed)
# setup sweep (what gets changed each run (seed, model, dataset, splits))
# log plots and metrics for each sweep (wandb.log, wandb.plot)
# have a gloabl plot from all runs which tracks mean, min, max

# See cifar100_classes
super_to_sub_str = '''aquatic mammals: beaver, dolphin, otter, seal, whale
fish: aquarium fish, flatfish, ray, shark, trout
flowers: orchids, poppies, roses, sunflowers, tulips
food containers: bottles, bowls, cans, cups, plates
fruit and vegetables: apples, mushrooms, oranges, pears, sweet peppers
household electrical devices: clock, computer keyboard, lamp, telephone, television
household furniture: bed, chair, couch, table, wardrobe
insects: bee, beetle, butterfly, caterpillar, cockroach
large carnivores: bear, leopard, lion, tiger, wolf
large man-made outdoor things: bridge, castle, house, road, skyscraper
large natural outdoor scenes: cloud, forest, mountain, plain, sea
large omnivores and herbivores: camel, cattle, chimpanzee, elephant, kangaroo
medium-sized mammals: fox, porcupine, possum, raccoon, skunk
non-insect invertebrates: crab, lobster, snail, spider, worm
people: baby, boy, girl, man, woman
reptiles: crocodile, dinosaur, lizard, snake, turtle
small mammals: hamster, mouse, rabbit, shrew, squirrel
trees: maple, oak, palm, pine, willow
vehicles 1: bicycle, bus, motorcycle, pickup truck, train
vehicles 2: lawn-mower, rocket, streetcar, tank, tractor'''

super_to_sub_map = {
    0: ([72, 4, 95, 30, 55], [1]),   # superclass -> [subclasses], [similar superclasses]
    1: ([73, 32, 67, 91, 1], [0]),
    2: ([92, 70, 82, 54, 62], [10, 17]),
    3: ([16, 61, 9, 10, 28], [5, 6]),
    4: ([51, 0, 53, 57, 83], []),
    5: ([40, 39, 22, 87, 86], [3, 6]),
    6: ([20, 25, 94, 84, 5], [3, 5]),
    7: ([14, 24, 6, 7, 18], [13]),
    8: ([43, 97, 42, 3, 88], [11, 12, 15, 16]),
    9: ([37, 17, 76, 12, 68], []),
    10: ([49, 33, 71, 23, 60], [2, 17]),
    11: ([15, 21, 19, 31, 38], [8, 12, 15, 16]),
    12: ([75, 63, 66, 64, 34], [8, 11, 15, 16]),
    13: ([77, 26, 45, 99, 79], [7]), 
    14: ([11, 2, 35, 46, 98], []),
    15: ([29, 93, 27, 78, 44], [8, 11, 12, 16]),
    16: ([65, 50, 74, 36, 80], [8, 11, 12, 15]),
    17: ([56, 52, 47, 59, 96], [2, 10]),
    18: ([8, 58, 90, 13, 48], [19]),
    19: ([81, 69, 41, 89, 85], [18])
}

def get_model(model_name, num_tasks, num_classes_each_task, in_channels=3):
    if model_name.lower() == 'resnet':
        model = ResNet(num_tasks=num_tasks, classes_each_task=num_classes_each_task, in_channels=in_channels)
    elif 'cnn' in model_name.lower():
        model = SemanticCNN(num_tasks=num_tasks, classes_each_task=num_classes_each_task, in_channels=in_channels)
    else:
        raise Exception('Need to test an available model')
    return model

def get_optimizer(optimizer_name, learning_rate=1e-3):
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    return optimizer

def train(model, train_set, optimizer, task, label_mapping, device="cpu"):
    model.train()
    correct = 0
    train_loss = 0
    for x_batch, y_batch in train_set:
        x_batch = x_batch.to(device)
        optimizer.zero_grad()

        output = model(x_batch, task)

        y_batch = torch.tensor([label_mapping[label.item()] for label in y_batch]).to(device)
        loss = nn.functional.cross_entropy(output, y_batch)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(y_batch.view_as(pred)).sum().item()

    train_loss = train_loss / ((len(train_set) - 1) * train_set[0][0].shape[0] + train_set[-1][0].shape[0])
    train_acc =  correct / ((len(train_set) - 1) * train_set[0][0].shape[0] + train_set[-1][0].shape[0])
    return {
        "loss": train_loss,
        "accuracy": train_acc
    }

def test(model, test_set, label_mapping, task=0, device="cpu"):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for x, y in test_set:
            x = x.to(device)
            output = model(x, task)

            y = torch.tensor([label_mapping[label.item()] for label in y]).to(device)
            test_loss += nn.functional.cross_entropy(output, y).item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss = test_loss / ((len(train_set) - 1) * train_set[0][0].shape[0] + train_set[-1][0].shape[0])
    test_acc = correct / ((len(test_set) - 1) * test_set[0][0].shape[0] + test_set[-1][0].shape[0])
    return {
        'loss': test_loss,
        'accuracy': test_acc
    }

def analyze_layers(pre_model, post_model, test_set, label_mapping, device, similarity, method='top-down'):
    if 'CNN' in str(type(pre_model)):
        conv_layers_pre = deepcopy(pre_model.conv_blocks[::3])
        conv_layers_post = post_model.conv_blocks[::3]
    elif 'ResNet' in str(type(pre_model)):
        conv_layers_pre = deepcopy(pre_model.res_layers)
        conv_layers_post = post_model.res_layers

    acc = []
    if method == 'top-down':
        # reset layers in top down fashion, evaluate on test set, log on wandb
        for i in range(len(conv_layers_post))[::-1]:
            conv_layers_post[i] = conv_layers_pre[i]
            test_metrics = test(post_model, test_set, label_mapping, device=device)
            test_acc = test_metrics['accuracy']
            acc.append(test_acc)
            wandb.log({
                f'layers/{method}-{similarity}/acc': test_acc
            })
    elif method == 'bottom-up':
        for i in range(len(conv_layers_post)):
            conv_layers_post[i] = conv_layers_pre[i]
            test_metrics = test(post_model, test_set, label_mapping, device=device)
            test_acc = test_metrics['accuracy']
            acc.append(test_acc)
            wandb.log({
                f'layers/{method}-{similarity}/acc': test_acc
            })
    
    return acc
    
################ main ###################

if __name__ == '__main__':
    # Get config
    config = get_config()
    locals().update(config)
    print(config)

    # Intialize wandb
    wandb.init(
        project=f"semantics-dissimilar-{dataset}-{model_name}",
        entity="continual-learning-2024",
        config=config
    )
    config.update(wandb.config)

    # Define task. Get train and test sets
    if dataset.lower() == 'cifar100':
        train_set, test_set = get_cifar('100')

        task1 = super_to_sub_map[starting_superclass][0]

        similar_classes = super_to_sub_map[starting_superclass][1]
        sim_idx = random.choice(similar_classes)
        task2_similar = super_to_sub_map[sim_idx][0]
        
        similar_classes.append(starting_superclass) # don't want to include initial task as dissimilar to itself
        dissimilar_classes = [i for i in range(len(super_to_sub_map)) if i not in similar_classes]
        diff_idx = random.choice(dissimilar_classes)
        task2_dissimilar = super_to_sub_map[diff_idx][0]
    else:
        # TODO add support for cifar10
        # Currently just the experiment from the graph in Ramasesh et. al. 2020
        train_set, test_set = get_cifar()
        task1 = [8,9]   # ship, truck
        task2_similar = [0,1]   # plane, car
        task2_dissimilar = [3,7]   # cat, horse

    data_split = split_data(dataset.lower(), train_set, test_set, custom_split=[task1, task2_similar, task2_dissimilar], batch_size=batch_size)

    # Set up training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(model_name, 2, [len(task1), len(task2_similar)])
    model.to(device)

    optimizer = get_optimizer(optimizer_name, learning_rate)

    epochs = 40 if dataset.lower() == 'cifar100' else 25

    ###################
    # Train on task 1 #
    ###################
    task = 0
    train_set = data_split[task]["train"]
    test_set = data_split[task]["test"]
    label_mapping1 = dict([(y, i) for i, y in enumerate(task1)])

    task1_acc_before = []   # running acc before task 2 for final plot
    for epoch in range(epochs):
        train_metrics = train(model, train_set, optimizer, task=task, label_mapping=label_mapping1, device=device)
        test_metrics = test(model, test_set, label_mapping=label_mapping1, device=device)

        task1_acc_before.append(test_metrics['accuracy'])

        # log metrics to wandb
        wandb.log({
            f'train/task1/loss': train_metrics['loss'],
            f'train/task1/acc': train_metrics['accuracy'],
            f'test/sim-acc': test_metrics['accuracy'],
            f'test/diff-acc': test_metrics['accuracy']
        })

        # Reshuffle
        train_set = reshuffle(train_set, batch_size) 
    
    # Save model for reuse
    torch.save(model.state_dict(), f"./{model_name}.pt")

    del train_set
    del test_set
    del model
    torch.cuda.empty_cache()

    #############################
    # Train on task 2 (Similar) #
    #############################
    task = 1
    train_set = data_split[task]["train"]
    test_set = data_split[0]['test']    # always test task 1
    label_mapping2 = dict([(y, i) for i, y in enumerate(task2_similar)])

    pre_model = get_model(model_name, 2, [len(task1), len(task2_similar)])
    pre_model.load_state_dict(torch.load(f"./{model_name}.pt"))
    model = deepcopy(pre_model)

    # Initialize lazy layers to allow hook
    dummy_tensor = torch.randn(train_set[0][0].shape)
    model(dummy_tensor, 0)
    model(dummy_tensor, 1)
    
    pre_model.to(device)
    model.to(device)
    wandb.watch(model, log='gradients', log_freq=50)

    optimizer = get_optimizer(optimizer_name, learning_rate)

    task1_acc_after_sim = []
    for epoch in range(epochs):
        train_metrics = train(model, train_set, optimizer, task=task, label_mapping=label_mapping2, device=device)
        test_metrics = test(model, test_set, label_mapping=label_mapping1, device=device)

        task1_acc_after_sim.append(test_metrics['accuracy'])

        wandb.log({
            f'train/task2-sim/loss': train_metrics['loss'],
            f'train/task2-sim/acc': train_metrics['accuracy'],
            f'test/sim-acc': test_metrics['accuracy'],
        })
        
        train_set = reshuffle(train_set, batch_size)
    
    wandb.unwatch(model)

    # Layer resets
    model_copy = deepcopy(model)
    acc_top_down = analyze_layers(pre_model, model, test_set, label_mapping1, device=device, similarity='sim', method='top-down')   # new model is destroyed
    acc_bottom_up = analyze_layers(pre_model, model_copy, test_set, label_mapping1, device=device, similarity='sim', method='bottom-up')

    plt.plot(range(len(acc_top_down)), acc_top_down, label=f"Top Down")
    plt.plot(range(len(acc_bottom_up)), acc_bottom_up, label=f"Bottom Up")
    plt.xlabel("Contiguous Layers Reset")
    plt.ylabel("Task 1 Accuracy")
    plt.title(f'N Layer Reset Experiment (Similar)')
    plt.legend()

    wandb.log({'reset-layers-sim': wandb.Image(plt)})
    plt.clf()

    del train_set
    del test_set
    del pre_model
    del model_copy
    del model
    torch.cuda.empty_cache()

    ################################
    # Train on task 2 (Dissimilar) #
    ################################
    task = 1
    train_set = data_split[task+1]["train"]
    test_set = data_split[0]['test']    # always test task 1
    label_mapping2 = dict([(y, i) for i, y in enumerate(task2_dissimilar)])

    pre_model = get_model(model_name, 2, [len(task1), len(task2_similar)])
    pre_model.load_state_dict(torch.load(f"./{model_name}.pt"))
    model = deepcopy(pre_model)

    # Initialize lazy layers to allow hook
    dummy_tensor = torch.randn(train_set[0][0].shape)
    model(dummy_tensor, 0)
    model(dummy_tensor, 1)

    pre_model.to(device)
    model.to(device)
    wandb.watch(model, log='gradients', log_freq=50)

    optimizer = get_optimizer(optimizer_name, learning_rate)

    task1_acc_after_diff = []
    for epoch in range(epochs):
        train_metrics = train(model, train_set, optimizer, task=task, label_mapping=label_mapping2, device=device)
        test_metrics = test(model, test_set, label_mapping=label_mapping1, device=device)

        task1_acc_after_diff.append(test_metrics['accuracy'])

        wandb.log({
            f'train/task2-diff/loss': train_metrics['loss'],
            f'train/task2-diff/acc': train_metrics['accuracy'],
            f'test/diff-acc': test_metrics['accuracy'],
        })
        
        train_set = reshuffle(train_set, batch_size)

    wandb.unwatch(model)

    # Reset layers experiment
    model_copy = deepcopy(model)
    acc_top_down = analyze_layers(pre_model, model, test_set, label_mapping1, device=device, similarity='diff', method='top-down')
    acc_bottom_up = analyze_layers(pre_model, model_copy, test_set, label_mapping1, device=device, similarity='diff', method='bottom-up')

    plt.plot(range(len(acc_top_down)), acc_top_down, label=f"Top Down")
    plt.plot(range(len(acc_bottom_up)), acc_bottom_up, label=f"Bottom Up")
    plt.xlabel("Contiguous Layers Reset")
    plt.ylabel("Task 1 Accuracy")
    plt.title(f'N Layer Reset Experiment (Dissimilar)')
    plt.legend()

    wandb.log({'reset-layers-diff': wandb.Image(plt)})
    plt.clf()
    
    # Plot the curves (in case I messed up wandb, and just to look nicer for reports)
    task1_acc_sim = task1_acc_before + task1_acc_after_sim
    task1_acc_diff = task1_acc_before + task1_acc_after_diff

    task1_superclass = super_to_sub_str.split('\n')[starting_superclass].split(':')[0]
    task2_sim_superclass = super_to_sub_str.split('\n')[sim_idx].split(':')[0]
    task2_diff_superclass = super_to_sub_str.split('\n')[diff_idx].split(':')[0]

    plt.plot(range(len(task1_acc_sim)), task1_acc_sim, label=f"Task 2: {task2_sim_superclass}")
    plt.plot(range(len(task1_acc_diff)), task1_acc_diff, label=f"Task 2: {task2_diff_superclass}")
    plt.xlabel("Epoch")
    plt.ylabel("Task 1 Accuracy")
    plt.title(f'Task 1: {task1_superclass}')
    plt.legend()

    wandb.log({'acc-curves': wandb.Image(plt)})

    # Free memory for next part of sweep
    gc.collect()
    torch.cuda.empty_cache()

