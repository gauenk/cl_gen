
import numpy as np
import torch,torchvision
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):

    def __init__(self, n_channels=1, embedding_size = 256, verbose = False ):
        super(Decoder, self).__init__()

        self.embedding_size = embedding_size
        self.i_size = int(np.sqrt(embedding_size))
        self.conv1 = SingleConv(n_channels, 32, kernel_size=2,stride=1)
        self.conv2 = SingleConv(32, 64, 1)
        self.conv3 = SingleConv(64, 128, 1)
        self.conv4 = SingleConv(128, 256, 1)
        self.conv5 = SingleConv(256, 512, 1)
        # self.conv6 = SingleConv(256, 256, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        if self.embedding_size == 64:
            self.fc1 = nn.Linear(576, 1024)
        else:
            self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, 28**2)

    def forward(self, x):
        x = x.reshape(-1,1,self.i_size,self.i_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x,1)
        print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.reshape(-1,1,28,28)
        return x


class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels, padding=0, kernel_size=2, stride=2):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)

