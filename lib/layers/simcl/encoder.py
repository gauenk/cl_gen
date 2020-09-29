

import torch,torchvision
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, n_channels=1, embedding_size = 128,
                 use_bn = True, verbose = False ):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.conv1 = SingleConv(n_channels, 48, (1,1),3,1,use_bn=use_bn)
        self.conv2 = SingleConv(48, 48, (1,1),3,1,use_bn=use_bn)
        self.conv3 = SingleConv(48, 48, (1,1),3,1,use_bn=use_bn)
        self.conv4 = SingleConv(48, 48, (1,1),3,1,use_bn=use_bn)
        self.conv5 = SingleConv(48, 48, (1,1),3,1,use_bn=use_bn,use_relu=False)
        # self.conv6 = SingleConv(48, 48, (1,1),3,1)

    def forward(self, x):
        skips = [x]
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,kernel_size=(2,2))
        skips.append(x)
        x = self.conv3(x)
        x = F.max_pool2d(x,kernel_size=(2,2))
        skips.append(x)
        x = self.conv4(x)
        x = F.max_pool2d(x,kernel_size=(2,2))
        x = self.conv5(x)
        x = torch.flatten(x,1)

        return x,skips


class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels, padding=(0,1), kernel_size=3, stride=1,use_bn=True,use_relu=True):
        super().__init__()
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding)
        bn = nn.BatchNorm2d(out_channels)
        l_relu = nn.LeakyReLU(inplace=True,negative_slope=0.01)
        use_bn = False

        layers = [conv2d]
        if use_bn: layers.append(bn)
        if use_relu: layers.append(l_relu)
        self.single_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.single_conv(x)

