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