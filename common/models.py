import torch
import torch.nn as nn

# Define a simple MLP described in https://arxiv.org/pdf/1904.07734
class MLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=400, output_size=10, activation=nn.ReLU, no_softmax=False):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.activation = activation()
        self.fc1 = nn.Linear(input_size, hidden_size)   # Input layer to first hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)       # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size)        # Second hidden layer to output layer
        self.softmax = nn.Softmax(dim=1)     # Softmax layer for output probabilities
        if no_softmax:
            self.softmax = nn.Identity()

    def forward(self, x):
        x = x.view(-1, self.input_size)              # Flatten the input image
        x = self.activation(self.fc1(x))          # First hidden layer with ReLU
        x = self.activation(self.fc2(x))          # Second hidden layer with ReLU
        x = self.fc3(x)                      # Output layer
        x = self.softmax(x)                  # Apply softmax activation to output
        return x
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel, 32 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32 input, 64 output
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 64 input, 128 output
        self.fc1 = nn.Linear(128 * 3 * 3, 64)  # Flattened size is 128 * 3 * 3
        self.fc2 = nn.Linear(64, 10)  # 10 classes for MNIST
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # First conv + relu + pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Second conv + relu + pooling
        x = self.pool(torch.relu(self.conv3(x)))  # Third conv + relu + pooling
        x = x.view(-1, 128 * 3 * 3)  # Flatten
        x = torch.relu(self.fc1(x))  # Fully connected layer 1
        x = self.dropout(x)          # Dropout
        x = self.fc2(x)              # Fully connected layer 2 (output layer)
        return x

####################################
# 8-block ResNet (He et. al. 2015) #
####################################
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.skip = None

        # Downsampling x if needed
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        out += (x if self.skip is None else self.skip(x))
        out = nn.functional.relu(out)
        return out

# This model has multi-head capabilities. 
# To use a single head, set num_tasks = [1] (yes this is a little hacky, sorry)
# and set classes_each_task = [total number of classes] (e.g. [10] for CIFAR10)
class ResNet(nn.Module):
    def __init__(self, num_tasks, classes_each_task, in_channels):
        """
        Args:
            num_tasks (int): number of tasks that will be run (used to decide number of heads in multi-head setup)
            classes_each_task (list[int]): classes_each_task[i] = number of classes for task i (used to determine logit size)
            in_channels (int): the initial number of channels (since it can differ depending on dataset)
        """
        super().__init__()
        # Initial convolutional layer
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32)
        )
        # ResNet Backbone
        self.res_layers = nn.Sequential(
            # ResNet MegaBlock 1
            ResNetBlock(32, 32),
            ResNetBlock(32, 32),
            # ResNet MegaBlock 2
            ResNetBlock(32, 64),
            ResNetBlock(64, 64),
            # ResNet MegaBlock 3
            ResNetBlock(64, 128),
            ResNetBlock(128, 128),
            # ResNet MegaBlock 4
            ResNetBlock(128, 256),
            ResNetBlock(256, 256),
        )
        # CLF heads (one per task)
        self.heads = nn.ModuleList(
            [nn.LazyLinear(classes_each_task[i]) for i in range(num_tasks)]
        )

    def forward(self, x, task):
        x = self.init_conv(x)
        x = self.res_layers(x)
        x = x.flatten(1)
        out = self.heads[task](x)
        return out

# Basically just the CNN defined above but multihead for semantics
class SemanticCNN(nn.Module):
    def __init__(self, num_tasks, classes_each_task, in_channels):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.dropout = nn.Dropout(0.5)
        self.heads = nn.ModuleList(
            [nn.LazyLinear(classes_each_task[i]) for i in range(num_tasks)]
        )

    def forward(self, x, task):
        x = self.conv_blocks(x)
        x = x.flatten(1)
        x = self.dropout(x)
        out = self.heads[task](x)
        return out