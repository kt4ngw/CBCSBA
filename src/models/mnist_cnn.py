import torch
import torch.nn as nn
import torch.nn.functional as F


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=4, padding=0)
        # 一层全连接层
        self.fc1 = nn.Linear(7 * 7 * 10, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        tensor = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = x.view(-1, 7 * 7 * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


    def get_model_size(self, ):
        total_params = 0
        for name, param in Mnist_CNN().named_parameters():
            layer_params = param.numel()
            total_params += layer_params
            # print(f"{name}: {layer_params} parameters")
        total_params_kb = (total_params * 4) / 1024 / 1024
        return total_params_kb


if __name__ == '__main__':
    # Calculate and print parameters for each layer
    total_params = 0
    for name, param in Mnist_CNN().named_parameters():
        layer_params = param.numel()
        total_params += layer_params
        print(f"{name}: {layer_params} parameters")
    total_params_kb = (total_params * 4) / 1024 / 1024
    print(f"Total model parameters: {total_params_kb:.2f} MB")