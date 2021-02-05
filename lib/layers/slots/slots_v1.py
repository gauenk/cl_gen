
# -- python imports --
import pdb
import numpy as np
import numpy.random as npr
from einops import rearrange, repeat, reduce
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils as tv_utils
from slot_attention import SlotAttention

# -- project imports --
import settings
from layers.simcl import ClBlockLoss

# -- [local] project imports --
from .utils import color_quantize,color_dequantize,plot_patches

# -- 3d party imports --
from slots_attention import SlotsAttention

class SlotsAttention32(nn.Module):

    def __init__(self, ftr_model, d_model, im_size, input_patch, output_patch, n_frames, bw=False, denoise_model = None, batch_size = 8):
        super(TransformerNetwork32, self).__init__()
        """
        Transformer for denoising

        im_size : int : the length and width of the square input image

        """
        self.d_model = d_model
        self.ftr_model = ftr_model
        print(d_model*n_frames,d_model)
        self.denoise_model_preproc_a = nn.Conv2d(d_model*n_frames,d_model*n_frames,3,1,1)
        # self.denoise_model_preproc_b = nn.Conv2d(d_model*n_frames,3*n_frames,3,1,1)
        # self.denoise_model_preproc_b = nn.Conv2d(d_model*n_frames,3,3,1,1)
        self.denoise_model_preproc_b = nn.Conv2d(d_model,3,3,1,1)
        self.denoise_model = denoise_model
        self.std = 5./255
        nhead = 4 # 4
        num_enc_layers = 2 # 6
        num_dec_layers = 2 # 6
        dim_ff = 256 # 512
        dropout = 0.1
        xform_args = [d_model, nhead, num_enc_layers, num_dec_layers, dim_ff, dropout]
        self.perform = Performer(d_model,4,4) # 8,4
        
        # -- slot attention network --
        num_slots = 5
        dim = 0
        iters = 3
        self.slots_attn = SlotsAttention(num_slots,dim,iters)

        # -- shift-translate network --
        


        # self.xform = nn.Transformer(*xform_args)
        self.clusters = np.load(f"{settings.ROOT_PATH}/data/kmeans_centers.npy")

        # -- constrastiv learning loss --
        num_transforms = n_frames
        hyperparams = edict()
        hyperparams.temperature = 0.1
        self.simclr_loss = ClBlockLoss(hyperparams, num_transforms, batch_size)

        # kwargs = {'num_tokens':512,'max_seq_len':input_patch**2,
        #           'dim':512, 'depth':6,'heads':4,'causal':False,'cross_attend':False}
        # self.xform_enc = PerformerLM(**kwargs)
        # self.xform_dec = PerformerLM(**kwargs)

        nhead = 1
        vdim = 1 if bw else 3
        self.attn = nn.MultiheadAttention(d_model,nhead,dropout,vdim=vdim)
        self.input_patch,self.output_patch = input_patch,output_patch
        self.im_size = im_size

        ftr_size = (input_patch**2)*d_model
        img_size = (input_patch**2)*3
        in_channels = d_model * n_frames
        # in_channels = 2304
        # self.conv = nn.Sequential(*[nn.Conv2d(in_channels,3,1)])
        stride = input_patch // output_patch
        padding = 1 if stride > 1 else 1
        # self.conv = nn.Sequential(*[nn.Conv2d(in_channels,3,3,stride,padding)])

        # -- left settings --; reduce imsize by 2 pix ( 10 in ,8 out )
        # stride = 1
        # padding = 0
        # self.conv = nn.Sequential(*[nn.Conv2d(in_channels,3,3,stride,padding)])

        # -- conv settings; reduce imsize by Half -- ( 16 in, 8 out ) ( A in, A/2 out )
        # stride = 2
        # padding = 1
        # self.conv = nn.Sequential(*[nn.Conv2d(in_channels,3,3,stride,padding)])

        # -- conv settings; maintain imsize -- ( A in, A out )
        stride = 1
        padding = 1
        out_chann = 1 if bw else 3
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels,out_chann,3,stride,padding)])

        self.end_conv = nn.Sequential(*[nn.Conv2d(3,3,1),nn.LeakyReLU(),nn.Conv2d(3,3,1)])

        padding = (input_patch - output_patch) // 2
        # stride = im_size-input_patch
        stride = output_patch
        print(input_patch,padding,stride)
        self.unfold_input = nn.Unfold(input_patch,1,padding,stride)
        # self.linear = nn.Linear(ftr_size,img_size)


    def forward(self, inputs, outputs):
        
        
        
        
