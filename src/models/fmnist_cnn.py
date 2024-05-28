import torch
import torch.nn as nn
import torch.nn.functional as F

class FMnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7*7*64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor
    def get_model_size(self, ):
        total_params = 0
        for name, param in FMnist_CNN().named_parameters():
            layer_params = param.numel()
            total_params += layer_params
            # print(f"{name}: {layer_params} parameters")
        total_params_kb = (total_params * 4) / 1024 / 1024
        return total_params_kb



    #     self.conv = nn.Sequential(
    #         nn.Conv2d(1, 32, 5),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2, stride=2),
    #         #nn.Dropout(0.3),
    #         nn.Conv2d(32, 64, 5),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2, stride=2),
    #        # nn.Dropout(0.3)
    #     )
    #     self.fc = nn.Sequential(
    #         nn.Linear(64*4*4, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, 10)
    #     )
        
    # def forward(self, x):
    #     x = self.conv(x)
    #     x = x.view(-1, 64*4*4)
    #     x = self.fc(x)
    #     # x = nn.functional.normalize(x)
    #     return x

if __name__ == '__main__':
    # Calculate and print parameters for each layer
    total_params = 0
    for name, param in FMnist_CNN().named_parameters():
        layer_params = param.numel()
        total_params += layer_params
        print(f"{name}: {layer_params} parameters")
    total_params_kb = (total_params * 4) / 1024 / 1024
    print(f"Total model parameters: {total_params_kb:.2f} MB")