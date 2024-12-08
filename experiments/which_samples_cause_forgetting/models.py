import torch
import torch.nn as nn

class MODEL(nn.Module):
    def get_gradient_norms_layerwise(self):
        norms = []
        for name, param in self.named_parameters():
            if param.grad is not None:
                gradient_norm = param.grad.norm(2).item()
                norms.append((name, gradient_norm))
        return norms

   
class CIFAR_CNN_1(MODEL):
    def __init__(self):
        super(CIFAR_CNN_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 100)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CIFAR_CNN_2(MODEL):
    def __init__(self):
        super(CIFAR_CNN_2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 100)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class CIFAR_CNN_3(MODEL):
    def __init__(self):
        super(CIFAR_CNN_3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 100)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class CIFAR_CNN_4(MODEL):
    def __init__(self):
        super(CIFAR_CNN_4, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(24, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 100)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(-1, 32 * 2 * 2)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CIFAR_MLP_1(MODEL):
    def __init__(self):
        super(CIFAR_MLP_1, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 400)
        self.fc2 = nn.Linear(400, 100)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CIFAR_MLP_2(MODEL):
    def __init__(self):
        super(CIFAR_MLP_2, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 400)
        self.fc2 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, 100)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = torch.sigmoid(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class CIFAR_MLP_3(MODEL):
    def __init__(self):
        super(CIFAR_MLP_3, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 400)
        self.fc2 = nn.Linear(400, 100)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
