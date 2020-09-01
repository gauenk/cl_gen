

import torch,torchvision
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, n_channels=1, embedding_size = 128, verbose = False ):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.conv1 = SingleConv(n_channels, 32, kernel_size=2,stride=1)
        self.conv2 = SingleConv(32, 64, 1)
        self.conv3 = SingleConv(64, 128, 1)
        self.conv4 = SingleConv(128, 256, 1)
        self.conv5 = SingleConv(256, 512, 1)

        self.conv1 = SingleConv(n_channels, 32, kernel_size=2,stride=1)
        self.conv2 = SingleConv(32, 64, 1)
        self.conv3 = SingleConv(64, 128, 1)
        self.conv4 = SingleConv(128, 256, 1)
        self.conv5 = SingleConv(256, 256, 1)
        self.conv6 = SingleConv(256, 256, 1)


        self.dropout1 = nn.Dropout2d(0.5)
        if embedding_size == 64:
            self.fc1 = nn.Linear(4608, 256)
        else:
            self.fc1 = nn.Linear(2704, 256)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(256, self.embedding_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x,1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels, padding=0, kernel_size=3, stride=2):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)

