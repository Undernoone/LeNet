import torch
from torch import nn

# define network
class LeNet(nn.Module):
    # initialize the network
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        # (28-5+2*2)/1 + 1 = 28
        self.sigmoid1 = nn.Sigmoid()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # (28-2)/2 + 1 = 14
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.output = nn.Linear(in_features=84, out_features=10)

    # forward pass
    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.output(x)
        return x

if __name__ == '__main__':
    x = torch.randn(1, 1, 28, 28)
    model = LeNet()
    y = model(x)
    print(y.shape)