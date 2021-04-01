
# -- python imports --
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvF

# -- project imports --

# -- 3d party imports --
from performer_pytorch import Performer,PerformerLM,FastAttention
from performer_pytorch.performer_pytorch import FeedForward,Chunk,PreLayerNorm,PreScaleNorm


class AttnBYOL(nn.Module):

    def __init__(self, num_frames, num_ftrs, patchsize, nh_size, img_size, attn_params = None):
        super().__init__()

        # -- init --
        self.num_frames = num_frames
        self.num_ftrs = num_ftrs
        self.nh_size = nh_size # number of patches around center pixel
        self._patchsize = patchsize
        self.img_size = img_size

        if attn_params is None:
            attn_params = edict()
            attn_params.d_model = 81*3
            attn_params.n_heads = 3
            attn_params.n_enc_layers = 2
            attn_params.n_dec_layers = 2
        self.attn_params = attn_params

        # -- create model --
        self.performer = PerformerEncDec(n_enc_layers=attn_params.n_enc_layers,
                                         n_dec_layers=attn_params.n_dec_layers,
                                         d_model=attn_params.d_model,
                                         n_heads=attn_params.n_heads)
        # self.perform = Performer(attn_params.d_model,
        #                          attn_params.n_heads,
        #                          attn_params.n_layers)

    def forward(self,patches):
        """
        params:
           [patches] shape: (LNB,C,H,W)
        """

        # -- init shapes --
        LNB,F = patches.shape[0],self.num_ftrs
        L,N = self.nh_size**2,2#self.num_frames
        B = LNB // (L*N)

        # -- BYOL init coniditon --
        if LNB != L*N*B: return torch.zeros(B,F)

        # -- forward pass starts here --
        patches = rearrange(patches,'(l n b) c h w -> n b l (c h w)',l=L,n=N)

        n_heads = self.attn_params.n_heads
        inputs = rearrange(patches[0],'b l (h f) -> b h l f',h=n_heads)
        outputs = rearrange(patches[1],'b l (h f) -> b h l f',h=n_heads)

        embeddings = self.performer(inputs,outputs)
        embeddings = rearrange(embeddings,'b l hf -> (b l) hf')

        print("emb",embeddings.shape,LNB,B,F)

        return embeddings


class PerformerEncDec(nn.Module):

    def __init__(self,n_enc_layers=2,n_dec_layers=2,n_heads=3,d_model=512,
                 ff_chunks=1,ff_mult=4,ff_dropout=0,ff_glu=False):
        super().__init__()

        # -- init vars --
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.n_heads = n_heads

        # -- create encoder layers --
        dim_heads = d_model//n_heads
        self.enc_attn,self.enc_ff = [],[]
        for _ in range(n_enc_layers):

            # -- attention layer --
            f_attn = FastAttention(dim_heads,d_model)
            self.enc_attn.append(f_attn)

            # -- post-proc layer --
            ff = FeedForward(d_model, mult = ff_mult,
                             dropout = ff_dropout, glu = ff_glu)
            chunk = Chunk( ff_chunks, ff, along_dim = 1)
            self.enc_ff.append(chunk)
        self.enc_attn = nn.Sequential(*self.enc_attn)
        self.enc_ff = nn.Sequential(*self.enc_ff)


        # -- create decoder layers --
        self.dec_1_attn,self.dec_2_attn,self.dec_ff = [],[],[]
        for _ in range(n_dec_layers):

            # -- attention layer --
            f_attn = FastAttention(dim_heads,d_model)
            self.dec_1_attn.append(f_attn)

            # -- attention layer --
            f_attn = FastAttention(dim_heads,d_model)
            self.dec_2_attn.append(f_attn)

            # -- post-proc layer --
            ff = FeedForward(d_model, mult = ff_mult,
                             dropout = ff_dropout, glu = ff_glu)
            chunk = Chunk( ff_chunks, ff, along_dim = 1)
            self.dec_ff.append(chunk)
        self.dec_1_attn = nn.Sequential(*self.dec_1_attn)
        self.dec_2_attn = nn.Sequential(*self.dec_2_attn)
        self.dec_ff = nn.Sequential(*self.dec_ff)

    def norm(self,to_norm,start=True,end=True):
        if start: to_norm = rearrange(to_norm,'b h l f -> b l (h f)')
        normed = self.layer_norm(to_norm)
        if end: normed = rearrange(normed,'b l (h f) -> b h l f',h=self.n_heads)
        return normed

    def forward(self,inputs,outputs):

        # -- extract layers --
        enc_attn_layers = list(self.enc_attn.children())
        enc_ff_layers = list(self.enc_ff.children())
        dec_1_attn_layers = list(self.dec_1_attn.children())
        dec_2_attn_layers = list(self.dec_2_attn.children())
        dec_ff_layers = list(self.dec_ff.children())

        # -- setup io --
        l_input,l_output = inputs,outputs

        # -- encoder --
        for layer_idx in range(self.n_enc_layers):

            # -- access layers --
            attn_l = enc_attn_layers[layer_idx]
            ff_l = enc_ff_layers[layer_idx]

            # -- encoder layer --
            attn_output = attn_l(l_input, l_input, l_input)
            ff_input = self.norm(attn_output + l_input,True,False)
            ff_output = ff_l(ff_input)
            l_input = self.norm(ff_output + ff_input,False,True)

        # -- dencoder --
        for layer_idx in range(self.n_dec_layers):

            # -- access layers --
            attn_1_l = dec_1_attn_layers[layer_idx]
            attn_2_l = dec_2_attn_layers[layer_idx]
            ff_l = dec_ff_layers[layer_idx]

            # -- decoder layer --
            attn_1_output = attn_1_l(l_output,l_output,l_output)
            attn_1_output = self.norm( attn_1_output + l_output, True, True)

            attn_2_output = attn_2_l(l_input, l_input, attn_1_output)
            attn_2_output = self.norm( attn_2_output + attn_1_output, True, False )

            l_output = self.norm( ff_l(attn_2_output) + attn_2_output , False, True)

        embeddings = rearrange(l_output,'b h l f -> b l (h f)')
        return embeddings
        


