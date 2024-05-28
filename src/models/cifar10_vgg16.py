
import torch
import torch.nn as nn
import torch.nn.functional as func


vgg = [96, 96, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class CIFAR10_VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10_VGG16, self).__init__()

        self.features = self._make_layers(vgg)
        self.dense = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Linear(4096, 10)
 
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        out = self.classifier(out)
        return out
 
    def _make_layers(self, vgg):
        layers = []
        in_channels = 3
        for x in vgg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
 
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

if __name__ == '__main__':
    model = CIFAR10_VGG16()
    use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
    print(use_gpu)
    print(model.cuda())
    # Calculate and print parameters for each layer
    total_params = 0
    for name, param in CIFAR10_VGG16().named_parameters():
        layer_params = param.numel()
        total_params += layer_params
        print(f"{name}: {layer_params} parameters")
    total_params_kb = (total_params * 4) / 1024 / 1024
    print(f"Total model parameters: {total_params_kb:.2f} MB")