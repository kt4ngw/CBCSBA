from torch import nn
import torch.nn.functional as F

class EMNISTCNN(nn.Module):
    def __init__(self,):
        super(EMNISTCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=40, kernel_size=5, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=160, kernel_size=5, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fcon1 = nn.Sequential(nn.Linear(49 * 160, 200), nn.LeakyReLU())
        self.fcon2 = nn.Linear(200, 47)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fcon1(x))
        x = self.fcon2(x)
        return x

    def get_model_size(self, ):
        total_params = 0
        for name, param in EMNISTCNN().named_parameters():
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
    for name, param in EMNISTCNN().named_parameters():
        layer_params = param.numel()
        total_params += layer_params
        print(f"{name}: {layer_params} parameters")
    total_params_kb = (total_params * 4) / 1024 / 1024
    print(f"Total model parameters: {total_params_kb:.2f} MB")