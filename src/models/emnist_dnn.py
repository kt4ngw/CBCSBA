import torch
import torch.nn as nn
import torch.nn.functional as F


class EMNISTDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 47)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return x

    def get_model_size(self, ):
        total_params = 0
        for name, param in EMNISTDNN().named_parameters():
            layer_params = param.numel()
            total_params += layer_params
            # print(f"{name}: {layer_params} parameters")
        total_params_kb = (total_params * 4) / 1024 / 1024
        return total_params_kb


if __name__ == '__main__':
    # Calculate and print parameters for each layer
    total_params = 0
    for name, param in EMNISTDNN().named_parameters():
        layer_params = param.numel()
        total_params += layer_params
        print(f"{name}: {layer_params} parameters")
    total_params_kb = (total_params * 4) / 1024 / 1024
    print(f"Total model parameters: {total_params_kb:.2f} MB")