
import numpy as np
import torch,torchvision
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):

    def __init__(self, n_channels=1, embedding_size = 256,
                 use_bn = True, verbose = False ):
        super(Decoder, self).__init__()
        self.n_channels = n_channels
        self.embedding_size = embedding_size
        self.i_size = int(np.sqrt(embedding_size))

        # self.upconv5 = SingleUpConv(48,48,0,2,2)
        # self.conv5a = SingleConv(96,96,(1,3),3,2)
        # self.conv5b = SingleConv(96,48,(1,2),3,1)

        self.upconv4 = SingleUpConv(48,48,0,2,2,use_bn=use_bn)
        self.conv4a = SingleConv(96,96,(1,1),3,1,use_bn=use_bn)
        self.conv4b = SingleConv(96,48,(1,1),3,1,use_bn=use_bn)

        self.upconv3 = SingleUpConv(48,48,use_bn=use_bn)
        self.conv3a = SingleConv(96,96,use_bn=use_bn)
        self.conv3b = SingleConv(96,48,use_bn=use_bn)

        self.upconv2 = SingleUpConv(48,48,use_bn=use_bn)
        self.conv2a = SingleConv(48+n_channels,64,use_bn=use_bn)
        self.conv2b = SingleConv(64,n_channels,use_bn=use_bn)

        # self.upconv1 = SingleUpConv(48,48)
        # self.conv1a = SingleConv(96,64)
        self.conv1a = SingleConv(n_channels,n_channels,0,1,1,False,False)
        self.conv1b = SingleConv(n_channels,n_channels,0,1,1,False,False)


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
        x = self.conv1a(x)
        x = self.conv1b(x)
        return x


class SingleUpConv(nn.Module):

    def __init__(self,in_channels,out_channels,padding=0,kernel_size=2,stride=2,use_bn=True,use_relu=True):
        super().__init__()
        conv2d = nn.ConvTranspose2d(in_channels , out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding)
        use_bn = False
        bn = nn.BatchNorm2d(out_channels)
        l_relu = nn.LeakyReLU(inplace=True,negative_slope=0.01)
        layers = [conv2d]
        if use_bn: layers.append(bn)
        if use_relu: layers.append(l_relu)
        self.up = nn.Sequential(*layers)

    def forward(self,x):
        return self.up(x)

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels, padding=(1,1), kernel_size=3, stride=1,use_bn=True,use_relu=True):
        super().__init__()
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding)
        use_bn = False
        bn = nn.BatchNorm2d(out_channels)
        l_relu = nn.LeakyReLU(inplace=True,negative_slope=0.01)

        layers = [conv2d]
        if use_bn: layers.append(bn)
        if use_relu: layers.append(l_relu)

        self.single_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.single_conv(x)

