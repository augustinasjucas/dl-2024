
import random

import torch

def batch(data, labels, batch_size):
    """
    Takes in a dataset (data, labels), batches it, randomly shuffles, and zips it
    Essentially, it's Dataloader but you can index any sample you want since it's a tensor
    This is good for things like gradeint analysis
    NOTE: this doesn't reshuffle the data, so call reshuffle(dataset, batch_size) each epoch!
    """
    data_batched = []
    labels_batched = []
    for i in range(0, len(data), batch_size):
        x_batch, y_batch = data[i:i+batch_size], labels[i:i+batch_size]
        permuted_idx = torch.randperm(len(x_batch))
        x_batch, y_batch = x_batch[permuted_idx], y_batch[permuted_idx]
        data_batched.append(x_batch)
        labels_batched.append(y_batch)
    dataset_batched = list(zip(data_batched, labels_batched))
    return dataset_batched


def reshuffle(batched_dataset, batch_size):
    """
    Takes in a batched dataset (batched_data, batched_labels), unzips it, and rebatches it
    """
    batched_data, batched_labels = zip(*batched_dataset)
    data, labels = torch.cat(batched_data), torch.cat(batched_labels)
    return batch(data, labels, batch_size)


def split_data(train, test, n_tasks=5, n_classes=10, custom_split=None, random_split=False, batch_size=32):
    """
    Splits train and test set into tasks as follows
        1. If `n_tasks` and `n_classes` is specified, it splits `n_classes` as evenly as possible among `n_tasks`, 
            doing so randomly if random_split is True
        2. If `custom_split` is defined, this becomes the task split
    
    Args:
      train: (dataset) training set, labels
      test: (dataset) test set, labels
      n_tasks: (int) number of tasks
      n_classes: (int) number of classes in dataset. Ensure n_classes % n_tasks == 0, otherwise, define a custom_split
      custom_split: (list[list[int]]) if not None, split of classes amongst tasks is done according to this list, with each sublist containing the classes for the task
      random_split: (bool) if True, split of classes amongst tasks is random
      batch_size: (int) batch size
    Returns:
      list[dict{'set': batched_set}] where
        index = task number
        set = 'train' or 'test'
        batched_set = batched dataset, accessible like dataloader (iterate over x_batch, y_batch)
    """
    if custom_split:
        tasks = custom_split
    else:
        classes = list(range(n_classes))
        if random_split:
            random.shuffle(classes)
        tasks = [classes[i:i+(n_classes // n_tasks)] for i in range(0, len(classes), n_classes // n_tasks)]


    x_train = torch.tensor(train.data).float()
    y_train = torch.tensor(train.targets)
    x_test = torch.tensor(test.data).float()
    y_test = torch.tensor(test.targets)

    data_split = []
    for task_labels in tasks:
        train_subset_inds = torch.isin(y_train, torch.tensor(task_labels))
        train_data = x_train[train_subset_inds]
        train_labels = y_train[train_subset_inds]
        train_set_batched = batch(train_data, train_labels, batch_size)

        test_subset_inds = torch.isin(y_test, torch.tensor(task_labels))
        test_data = x_test[test_subset_inds]
        test_labels = y_test[test_subset_inds]
        test_set_batched = batch(test_data, test_labels, batch_size)

        data_split.append({
            'train': train_set_batched,
            'test': test_set_batched
        })

    return data_split