import torch
from torch.utils.data import DataLoader

import wandb


def simple_test(
    model: torch.nn.Module, test_loader: DataLoader, criterion, device: str
):
    # Reports the average loss and accuracy of the model on the test set
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_pred = model(batch_x)

            test_loss += criterion(batch_pred, batch_y).item()
            pred = batch_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(batch_y.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    return test_loss, accuracy


def simple_train(
    model,
    train_loader,
    optimizer,
    criterion,
    epochs,
    device,
    wandb_log_path: str = None,
    test_loader=None,
):
    # Performs vanilla training on the model, and given data and all needed info

    model.train()

    for epoch_num in range(epochs):
        epoch_acc, epoch_loss = 0, 0
        for batch_num, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            batch_pred = model(batch_x)
            loss = criterion(batch_pred, batch_y)

            epoch_loss += loss.item()
            pred = batch_pred.argmax(dim=1, keepdim=True)
            epoch_acc += pred.eq(batch_y.view_as(pred)).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= len(train_loader.dataset)
        epoch_acc /= len(train_loader.dataset)

        if test_loader is not None:
            test_loss, test_acc = simple_test(model, test_loader, criterion, device)
            test_data = {
                f"{wandb_log_path}intermediate_test_loss": test_loss,
                f"{wandb_log_path}intermediate_test_accuracy": test_acc,
            }
        else:
            test_data = {}

        if wandb_log_path is not None:
            wandb.log(
                {
                    f"{wandb_log_path}intermediate_train_loss": epoch_loss,
                    f"{wandb_log_path}intermediate_train_accuracy": epoch_acc,
                    **test_data,
                }
            )
