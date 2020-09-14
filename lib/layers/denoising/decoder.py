
import numpy as np
import torch,torchvision
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):

    def __init__(self, n_channels=1, embedding_size = 256, verbose = False ):
        super(Decoder, self).__init__()
        self.n_channels = n_channels
        self.embedding_size = embedding_size
        self.i_size = int(np.sqrt(embedding_size))

        # self.upconv5 = SingleUpConv(48,48,0,2,2)
        # self.conv5a = SingleConv(96,96,(1,3),3,2)
        # self.conv5b = SingleConv(96,48,(1,2),3,1)

        self.upconv4 = SingleUpConv(48,48,0,2,2)
        self.conv4a = SingleConv(96,96,(1,1),3,1)
        self.conv4b = SingleConv(96,48,(1,1),3,1)

        self.upconv3 = SingleUpConv(48,48)
        self.conv3a = SingleConv(96,96)
        self.conv3b = SingleConv(96,48)

        self.upconv2 = SingleUpConv(48,48)
        self.conv2a = SingleConv(48+n_channels,64)
        self.conv2b = SingleConv(64,n_channels)

        # self.upconv1 = SingleUpConv(48,48)
        # self.conv1a = SingleConv(96,64)
        # self.conv1b = SingleConv(64,1)


    def forward(self, inputs):
        x,skips = inputs
        BS = x.shape[0]
        img_x = x
        x = x.reshape(BS,48,4,4)
        x = self.upconv4(x)
        a = skips.pop()
        x = torch.cat([x,a], dim=1)
        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.upconv3(x)
        a = skips.pop()
        x = torch.cat([x,a], dim=1)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.upconv2(x)
        a = skips.pop()
        x = torch.cat([x,a], dim=1)
        x = self.conv2a(x)
        x = self.conv2b(x)
        # if self.n_channels == 1:
        #     x = F.relu(x)
        return x


class SingleUpConv(nn.Module):

    def __init__(self,in_channels,out_channels,padding=0,kernel_size=2,stride=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels , out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True,negative_slope=0.01),
        )
    def forward(self,x):
        return self.up(x)

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels, padding=(1,1), kernel_size=3, stride=1):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True,negative_slope=0.01)
        )

    def forward(self, x):
        return self.single_conv(x)

