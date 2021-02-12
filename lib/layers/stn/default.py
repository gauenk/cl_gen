
# -- python import --
import numpy as np
from einops import rearrange, repeat, reduce


# -- pytorch imports --
import torch
from torch import nn
import torch.nn.functional as F


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        BS = x.shape[0]
        xs = self.localization(x)
        xs = xs.view(BS,-1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stn(x)
        return x

class Basic(nn.Module):
    def __init__(self, in_ch, out_ch, g=16, channel_att=False, spatial_att=False):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

        if channel_att:
            self.att_c = nn.Sequential(
                nn.Conv2d(2*out_ch, out_ch//g, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(out_ch//g, out_ch, 1, 1, 0),
                nn.Sigmoid()
            )
        if spatial_att:
            self.att_s = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

    def forward(self, data):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        fm = self.conv1(data)
        if self.channel_att:
            # fm_pool = F.adaptive_avg_pool2d(fm, (1, 1)) + F.adaptive_max_pool2d(fm, (1, 1))
            fm_pool = torch.cat([F.adaptive_avg_pool2d(fm, (1, 1)), F.adaptive_max_pool2d(fm, (1, 1))], dim=1)
            att = self.att_c(fm_pool)
            fm = fm * att
        if self.spatial_att:
            fm_pool = torch.cat([torch.mean(fm, dim=1, keepdim=True), torch.max(fm, dim=1, keepdim=True)[0]], dim=1)
            att = self.att_s(fm_pool)
            fm = fm * att
        return fm

class STNPairs(nn.Module):
    def __init__(self,img_shape,in_channels=3*2,frame_size=256):
        super(STNPairs, self).__init__()
        img_size = np.product(img_shape)
        
        # -- frame size --
        if frame_size == 256: dims = 10 * 8 * 8
        elif frame_size == 128: dims = 10 * 28 * 28
        elif frame_size == 64: dims = 10 * 12 * 12

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            Basic(in_channels, 64, channel_att=False, spatial_att=False),
            nn.MaxPool2d(2),
            Basic(64, 128, channel_att=False, spatial_att=False),
            nn.MaxPool2d(2),
            Basic(128, 256, channel_att=False, spatial_att=False),
            nn.MaxPool2d(2),
            Basic(256, 128, channel_att=False, spatial_att=False),
            nn.MaxPool2d(2),
            Basic(128, 64, channel_att=False, spatial_att=False),
            nn.MaxPool2d(2),
            Basic(64, 10, channel_att=False, spatial_att=False),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(dims, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, pair):
        """
        :params pair: [manipulated_image, refernece_image] shape: [B,2,C,H,W]
        """
        BS = pair.shape[0]
        target = pair[:,0]
        pair = rearrange(pair,'b n c h w -> b (n c) h w')

        xs = self.localization(pair)
        xs = xs.view(BS,-1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, target.size(),align_corners=True)
        x = F.grid_sample(target, grid, align_corners=True)

        return x,theta

    def forward(self, x):
        x,theta = self.stn(x)
        return x,theta



class STNBurst(nn.Module):
    def __init__(self,img_shape):
        super(STNBurst, self).__init__()
        self.pair_net = STNPairs(img_shape)

    def forward(self, burst):
        """
        :params burst: shape: [B,N,C,H,W]
        """
        B,N,C,H,W = burst.shape
        aligned = []
        thetas = []
        for i in range(N):
            if i == N//2:
                aligned.append(burst[:,N//2])
                continue
            pair = torch.stack([burst[:,i],burst[:,N//2]],dim=1)
            aligned_i,theta_i = self.pair_net(pair)
            aligned.append(aligned_i)
            thetas.append(theta_i)
        aligned = torch.stack(aligned,dim=1)
        thetas = torch.stack(thetas,dim=1)
        # print("stnburst",thetas)
        return aligned,thetas
