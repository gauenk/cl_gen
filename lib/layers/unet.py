""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels, verbose = False ):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.verbose = verbose

        self.conv1 = SingleConv(n_channels, 32, kernel_size=2,stride=1)
        self.conv2 = SingleConv(32, 64, 1)
        self.conv3 = SingleConv(64, 128, 1)
        self.conv4 = SingleConv(128, 256, 1)
        self.conv5 = SingleConv(256, 256, 1)
        self.conv6 = SingleConv(256, 256, 1)

        self.up1 = SingleUpConv(256,256)
        self.up2 = SingleUpConv(512,256)
        self.up3 = SingleUpConv(512,128)
        self.up4 = SingleUpConv(256,64)
        self.up5 = SingleUpConv(128,32)
        self.up6 = SingleUpConv(64,32,0,2,1)

        self.end1 = SingleConv(32,32, 1, 3, 1)
        self.end2 = SingleConv(32, 1, 0, 1, 1)

    def forward(self, x):
        self.vprint("fwd")
        self.vprint(x.shape)
        x1 = self.conv1(x)
        self.vprint(x1.shape)
        x2 = self.conv2(x1)
        self.vprint(x2.shape)
        x3 = self.conv3(x2)
        self.vprint(x3.shape)
        x4 = self.conv4(x3)
        self.vprint(x4.shape)
        x5 = self.conv5(x4)
        self.vprint(x5.shape)
        x6 = self.conv6(x5)
        self.vprint(x6.shape)
        
        u1 = self.up1(x6)
        self.vprint(u1.shape)
        u2 = self.up2(torch.cat([x5,u1],dim=1))
        self.vprint(u2.shape)
        u3 = self.up3(torch.cat([x4,u2],dim=1))
        self.vprint(u3.shape)
        u4 = self.up4(torch.cat([x3,u3],dim=1))
        self.vprint('u4',u4.shape)
        u5 = self.up5(torch.cat([x2,u4],dim=1))
        self.vprint('u5',u5.shape)
        u6 = self.up6(torch.cat([x1,u5],dim=1))
        self.vprint('u6',u6.shape)
        
        e1 = self.end1(u6)
        self.vprint("e1",e1.shape)

        e2 = self.end2(e1)
        self.vprint("e2",e2.shape)

        return e2

    def vprint(self,*msg):
        if self.verbose:
            print(*msg)


class UNet_n2n(nn.Module):
    def __init__(self, n_channels, verbose = False ):
        super(UNet_n2n, self).__init__()
        self.n_channels = n_channels
        self.verbose = False

        self.conv1 = DoubleConv(3*n_channels, 48, kernel_size=3, padding=1)
        self.conv2 = SingleConv(48, 48, 1)
        self.conv3 = SingleConv(48, 48, 1)
        self.conv4 = SingleConv(48, 48, 1)
        self.conv5 = SingleConv(48, 48, 1)
        self.conv6 = SingleConv(48, 48, 1, use_pool=False)

        self.up1 = Up(96,96)
        self.up2 = Up(144,96)
        self.up3 = Up(144,96)
        self.up4 = Up(144,96)
        self.up5 = Up(96+3*n_channels,96)
        
        self.out_conv = SingleConv(96,3,kernel_size=3,padding=1,use_pool=False,use_relu=False)

        # self.end1 = SingleConv(32,32, 1, 3, 1)
        # self.end2 = SingleConv(32, 1, 0, 1, 1)

    def forward(self, x):
        self.vprint("fwd")
        self.vprint('x',x.shape)
        x1 = self.conv1(x)
        self.vprint('x1',x1.shape)
        x2 = self.conv2(x1)
        self.vprint('x2',x2.shape)
        x3 = self.conv3(x2)
        self.vprint('x3',x3.shape)
        x4 = self.conv4(x3)
        self.vprint('x4',x4.shape)
        x5 = self.conv5(x4)
        self.vprint('x5',x5.shape)
        x6 = self.conv6(x5)
        self.vprint('x6',x6.shape)
        
        # u1 = self.up1(x6)
        # self.vprint(u1.shape)
        # u2 = self.up2(torch.cat([x5,u1],dim=1))
        # self.vprint(u2.shape)
        # u3 = self.up3(torch.cat([x4,u2],dim=1))
        # self.vprint(u3.shape)
        # u4 = self.up4(torch.cat([x3,u3],dim=1))
        # self.vprint('u4',u4.shape)
        # u5 = self.up5(torch.cat([x2,u4],dim=1))
        # self.vprint('u5',u5.shape)
        # u6 = self.up6(torch.cat([x1,u5],dim=1))
        # self.vprint('u6',u6.shape)
        
        u1 = self.up1(x6,x4)
        self.vprint('u1',u1.shape)
        u2 = self.up2(u1,x3)
        self.vprint('u2',u2.shape)
        u3 = self.up3(u2,x2)
        self.vprint('u3',u3.shape)
        u4 = self.up4(u3,x1)
        self.vprint('u4',u4.shape)
        u5 = self.up5(u2,x)
        self.vprint('u5',u5.shape)
        u6 = self.out_conv(u5)
        self.vprint('u6',u6.shape)

        return u6

    def vprint(self,*msg):
        if self.verbose:
            print(*msg)



#
# Parts of UNet Model
#


class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels, padding=0, kernel_size=3, stride=1, use_relu = True, use_pool=True):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding))
        layers.append(nn.BatchNorm2d(out_channels))
        if use_relu: layers.append(nn.ReLU(inplace=True))
        if use_pool: layers.append(nn.MaxPool2d(2))
        self.single_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.single_conv(x)
    

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, padding=1, stride=1, use_pool = True):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        layers = []
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, stride=stride))
        layers.append(nn.BatchNorm2d(mid_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)),
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if use_pool: layers.append(nn.MaxPool2d(2))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class SingleUpConv(nn.Module):

    def __init__(self,in_channels,out_channels,padding=1,kernel_size=3,stride=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , out_channels,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding)
    def forward(self,x):
        return self.up(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size=3, stride=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # self.up = nn.ConvTranspose2d(out_channels // 2 , out_channels,
            #                              kernel_size=kernel_size, stride=stride)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            self.conv = DoubleConv(in_channels, out_channels, None,use_pool=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2,
                                         kernel_size=kernel_size, stride=stride)
            self.conv = DoubleConv(in_channels, out_channels,use_pool=False)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print("up_x",x1.shape)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # print("up_x1_pad",x1.shape)
        x = torch.cat([x2, x1], dim=1)
        # print("up_xcat",x.shape)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet_v2(nn.Module):
    def __init__(self, n_channels, verbose = False ):
        super(UNet_v2, self).__init__()
        self.n_channels = n_channels
        self.verbose = verbose

        self.conv1 = SingleConv(n_channels, 32, kernel_size=2,stride=2)
        self.conv2 = SingleConv(32, 64, 1)
        self.conv3 = SingleConv(64, 128, 1)
        self.conv4 = SingleConv(128, 256, 1)
        self.conv5 = SingleConv(256, 256, 1)
        # self.conv6 = SingleConv(256, 256, 1)

        # self.up1 = SingleUpConv(256,256)
        self.up2 = SingleUpConv(256,256)
        self.up3 = SingleUpConv(512,128)
        self.up4 = SingleUpConv(256,64)
        self.up5 = SingleUpConv(128,32)
        self.up6 = SingleUpConv(64,32)

        self.end1 = SingleConv(32,32, 1, 3, 1)
        self.end2 = SingleConv(32, 3, 0, 1, 1,False)

    def forward(self, x):
        self.vprint("fwd")
        self.vprint('input',x.shape)
        x1 = self.conv1(x)
        self.vprint('x1',x1.shape)
        x2 = self.conv2(x1)
        self.vprint('x2',x2.shape)
        x3 = self.conv3(x2)
        self.vprint('x3',x3.shape)
        x4 = self.conv4(x3)
        self.vprint('x4',x4.shape)
        x5 = self.conv5(x4)
        self.vprint('x5',x5.shape)
        # x6 = self.conv6(x5)
        # self.vprint('x6',x6.shape)
        
        # u1 = self.up1(x6)
        # self.vprint(u1.shape)
        u2 = self.up2(x5)
        self.vprint('u2',u2.shape)
        u3 = self.up3(torch.cat([x4,u2],dim=1))
        self.vprint('u3',u3.shape)
        u4 = self.up4(torch.cat([x3,u3],dim=1))
        self.vprint('u4',u4.shape)
        u5 = self.up5(torch.cat([x2,u4],dim=1))
        self.vprint('u5',u5.shape)
        u6 = self.up6(torch.cat([x1,u5],dim=1))
        self.vprint('u6',u6.shape)
        
        e1 = self.end1(u6)
        self.vprint("e1",e1.shape)

        e2 = self.end2(e1)
        self.vprint("e2",e2.shape)

        return e2

    def vprint(self,*msg):
        if self.verbose:
            print(*msg)



class UNet_v2_with_noise(nn.Module):
    def __init__(self, n_channels, verbose = False ):
        super(UNet_v2_with_noise, self).__init__()
        self.n_channels = n_channels
        self.verbose = verbose

        self.conv1 = SingleConv(2*n_channels, 32, kernel_size=2,stride=2)
        self.conv2 = SingleConv(32, 64, 1)
        self.conv3 = SingleConv(64, 128, 1)
        self.conv4 = SingleConv(128, 256, 1)
        self.conv5 = SingleConv(256, 256, 1)
        # self.conv6 = SingleConv(256, 256, 1)

        # self.up1 = SingleUpConv(256,256)
        self.up2 = SingleUpConv(256,256)
        self.up3 = SingleUpConv(512,128)
        self.up4 = SingleUpConv(256,64)
        self.up5 = SingleUpConv(128,32)
        self.up6 = SingleUpConv(64,32)

        self.end1 = SingleConv(32,32, 1, 3, 1)
        self.end2 = SingleConv(32, 3, 0, 1, 1,False)

    def forward(self, img, delta):
        self.vprint("fwd")
        x = torch.cat([img,delta],dim=1)
        self.vprint('input',x.shape)
        x1 = self.conv1(x)
        self.vprint('x1',x1.shape)
        x2 = self.conv2(x1)
        self.vprint('x2',x2.shape)
        x3 = self.conv3(x2)
        self.vprint('x3',x3.shape)
        x4 = self.conv4(x3)
        self.vprint('x4',x4.shape)
        x5 = self.conv5(x4)
        self.vprint('x5',x5.shape)
        # x6 = self.conv6(x5)
        # self.vprint('x6',x6.shape)
        
        # u1 = self.up1(x6)
        # self.vprint(u1.shape)
        u2 = self.up2(x5)
        self.vprint('u2',u2.shape)
        u3 = self.up3(torch.cat([x4,u2],dim=1))
        self.vprint('u3',u3.shape)
        u4 = self.up4(torch.cat([x3,u3],dim=1))
        self.vprint('u4',u4.shape)
        u5 = self.up5(torch.cat([x2,u4],dim=1))
        self.vprint('u5',u5.shape)
        u6 = self.up6(torch.cat([x1,u5],dim=1))
        self.vprint('u6',u6.shape)
        
        e1 = self.end1(u6)
        self.vprint("e1",e1.shape)

        e2 = self.end2(e1)
        self.vprint("e2",e2.shape)

        return e2

    def vprint(self,*msg):
        if self.verbose:
            print(*msg)



class UNetN_v2(nn.Module):
    def __init__(self, N, n_channels, verbose = False ):
        super(UNetN_v2, self).__init__()
        self.n_channels = n_channels
        self.verbose = verbose
        # assert (N % 2) == 0, "N must be even."
        # N_half = N // 2

        self.conv1 = SingleConv(N*n_channels, 32, kernel_size=2,stride=2)
        self.conv2 = SingleConv(32, 64, 1)
        self.conv3 = SingleConv(64, 128, 1)
        self.conv4 = SingleConv(128, 256, 1)
        self.conv5 = SingleConv(256, 256, 1)
        # self.conv6 = SingleConv(256, 256, 1)

        # self.up1 = SingleUpConv(256,256)
        self.up2 = SingleUpConv(256,256)
        self.up3 = SingleUpConv(512,128)
        self.up4 = SingleUpConv(256,64)
        self.up5 = SingleUpConv(128,32)
        self.up6 = SingleUpConv(64,32)

        self.end1 = SingleConv(32,32, 1, 3, 1)
        self.end2 = SingleConv(32, n_channels, 0, 1, 1,False)

    def forward(self, x):
        self.vprint("fwd")
        self.vprint('input',x.shape)
        x1 = self.conv1(x)
        self.vprint('x1',x1.shape)
        x2 = self.conv2(x1)
        self.vprint('x2',x2.shape)
        x3 = self.conv3(x2)
        self.vprint('x3',x3.shape)
        x4 = self.conv4(x3)
        self.vprint('x4',x4.shape)
        x5 = self.conv5(x4)
        self.vprint('x5',x5.shape)
        # x6 = self.conv6(x5)
        # self.vprint('x6',x6.shape)
        
        # u1 = self.up1(x6)
        # self.vprint(u1.shape)
        u2 = self.up2(x5)
        self.vprint('u2',u2.shape)
        u3 = self.up3(torch.cat([x4,u2],dim=1))
        self.vprint('u3',u3.shape)
        u4 = self.up4(torch.cat([x3,u3],dim=1))
        self.vprint('u4',u4.shape)
        u5 = self.up5(torch.cat([x2,u4],dim=1))
        self.vprint('u5',u5.shape)
        u6 = self.up6(torch.cat([x1,u5],dim=1))
        self.vprint('u6',u6.shape)
        
        e1 = self.end1(u6)
        self.vprint("e1",e1.shape)

        e2 = self.end2(e1)
        self.vprint("e2",e2.shape)

        return e2

    def vprint(self,*msg):
        if self.verbose:
            print(*msg)

