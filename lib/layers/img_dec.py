import torch
import torch.nn.functional as F

from .unet import UNet

class ImageDecoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.network_1 = UNet(2)
        self.network_2 = UNet(3)

    def forward(self,x_set):
        # print(x_set[0].shape)
        i_shape = x_set[0].shape
        if len(i_shape) > 1: batch_size = i_shape[0]
        else: batch_size = 1
        dec_x = []
        for x in x_set:
            x = F.interpolate(x.reshape(batch_size,2,32,32),33)
            dec_x_i = self.single_forward(x)
            dec_x.append(dec_x_i)
        return dec_x

    def single_forward(self,input_1):
        output_1 = self.network_1(input_1)
        input_2 = torch.cat([input_1,output_1],dim=1)
        output_2 = self.network_2(input_2)
        return output_2
