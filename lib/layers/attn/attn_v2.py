
# -- python imports --
import pdb
from einops import rearrange, repeat, reduce
import numpy as np

# -- pytorch imports --
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils as tv_utils

# -- project imports --
import settings

# -- local project imports --
from .utils import color_quantize,color_dequantize

# -- 3d party imports --
from performer_pytorch import Performer,PerformerLM,FastAttention

class TransformerNetwork32_noxform(nn.Module):

    def __init__(self, ftr_model, d_model, im_size, input_patch, output_patch, n_frames, bw=False, denoise_model = None):
        super(TransformerNetwork32_noxform, self).__init__()
        """
        Transformer for denoising

        im_size : int : the length and width of the square input image

        """
        self.d_model = d_model
        self.ftr_model = ftr_model
        self.denoise_model_preproc = nn.Conv2d(d_model*n_frames,3*n_frames,3,1,1)
        self.denoise_model = denoise_model
        nhead = 4
        num_enc_layers = 6
        num_dec_layers = 6
        dim_ff = 512
        dropout = 0.1
        xform_args = [d_model, nhead, num_enc_layers, num_dec_layers, dim_ff, dropout]
        self.clusters = np.load(f"{settings.ROOT_PATH}/data/kmeans_centers.npy")


        # kwargs = {'num_tokens':512,'max_seq_len':input_patch**2,
        #           'dim':512, 'depth':6,'heads':4,'causal':False,'cross_attend':False}
        # self.xform_enc = PerformerLM(**kwargs)
        # self.xform_dec = PerformerLM(**kwargs)

        nhead = 1
        vdim = 1 if bw else 3
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
        self.unfold_input = nn.Unfold(input_patch,1,padding,stride)
        # self.linear = nn.Linear(ftr_size,img_size)

    def forward(self, inputs, outputs):
        """
        inputs (array-like): (BS,N,H,W,C)
        outputs (array-like): (BS,H,W,C)

        BS = batch-size
        N = number of frames
        H,W,C = dimensions of images
        
        -- nn.Transformer shapes --
        -> inputs (S,BS,E)
        -> targets (T,BS,E)
        -> outputs (T,BS,E)

        S = source sequence length
        T = target sequence length
        BS = batch size
        E = encoded dimension
        """
        # -- init variables --
        BS,N,C,H,W = inputs.shape
        i_p,o_p = self.input_patch,self.output_patch
        # m_index = N//2
        # no_mid = np.r_[np.arange(m_index),np.arange(m_index+1,N)]


        # -- feature extraction --
        fm_inputs = rearrange(inputs,'bs n c h w -> n bs c h w')
        if C == 1: fm_inputs = repeat(fm_inputs,'n bs 1 h w -> n bs c h w',c=3)
        # src_ftrs = self.ftr_model(fm_inputs).detach()
        # tgt_ftrs = self.ftr_model(rearrange(outputs,'bs c h w -> 1 bs c h w')).detach()[0]

        # -- each pixel value turns into indicator var --
        src_ftrs = color_quantize(fm_inputs,self.clusters,self.d_model)
        # qoutputs = color_quantize(outputs,self.clusters,self.d_model)
        
        # -- feature map to image size --
        # print(src_ftrs.shape,tgt_ftrs.shape)
        # src_ftrs = rearrange(src_ftrs,'n bs (c h w) -> n bs c h w',h=i_p,w=i_p)
        # tgt_ftrs = rearrange(tgt_ftrs,'1 bs (c h w) -> bs c h w',h=i_p,w=i_p)

        # -- extract patches --
        # nm_patches = self.unfold_input(src_ftrs[m_index])
        # nm_patches = rearrange(nm_patches,'b (c i) r -> r i b c',b=BS,i=i_p**2)
        # # src_ftrs = src_ftrs[no_mid]
        # nmi_patches = rearrange(self.unfold_input(inputs[:,m_index]),'(b n) (c i) r -> r (n i) b c',b=BS,i=i_p**2)

        in_patches = self.unfold_input(rearrange(inputs,'bs n c h w -> (bs n) c h w'))
        in_patches = rearrange(in_patches,'(b n) (c i) r -> r (n i) b c',b=BS,i=i_p**2)
        # in_patches = rearrange(inputs,'bs n c (rh h) (rw w) -> (rh rw) (n h w) bs c',h=i_p,w=i_p)
        out_patches = rearrange(outputs,'bs c (rh h) (rw w) -> (rh rw) bs c h w',h=o_p,w=o_p)
        src_patches = self.unfold_input(rearrange(src_ftrs,'n bs c h w -> (n bs) c h w'))
        src_patches = rearrange(src_patches,'(n b) (c i) r -> r (n i) b c',b=BS,i=i_p**2)
        # src_patches=rearrange(src_ftrs,'n bs c (rh h) (rw w) -> (rh rw) (n h w) bs c',h=i_p,w=i_p)
        # tgt_patches = rearrange(tgt_ftrs,'bs c (rh h) (rw w) -> (rh rw) (h w) bs c',h=i_p,w=i_p)
        assert in_patches.shape[0] == src_patches.shape[0], "Number of patches must match"
        assert in_patches.shape[0] == out_patches.shape[0], "Number of patches must match"
        n_patches = in_patches.shape[0]

        # -- compute transformer for each patch --; [todo] Patches concats with BS
        rec_image = []
        loss = 0
        for patch in range(n_patches):

            # -- xform inputs are SRC and SRC --
            # print(src_patches[patch].shape)
            # st_patch = src_patches[patch]
            # st_patch = self.xform(src_patches[patch],src_patches[patch])
            # st_patch = self.xform(st_patch,nm_patches[patch])
            # print(src_patches[patch].shape)
            # src_patches[patch] = src_patches[patch].type(torch.long)
            # st_patch = self.xform_enc(src_patches[patch][0].type(torch.long), return_encodings = True)
            # st_patch = self.xform_dec(in_patches[patch], context = st_patch)

            # -- final attention to "go back to pixel space"; don't need it for quant --
            # print(st_patch.shape)
            
            # rec_patch_p,_ = self.attn(st_patch,st_patch,in_patches[patch])
            # pdb.set_trace()
            # rec_patch_p,_ = self.attn(st_patch,st_patch,nm_patches[patch])
            # rec_patch_p,_ = self.attn(nm_patches[patch],st_patch,st_patch)
            # rec_patch_p = st_patch

            # -- conv to rec size --
            # rec_patch_p = rearrange(rec_patch_p,'(n h w) bs d -> bs (n d) h w',h=i_p,w=i_p) 
            # rec_patch = self.conv(rec_patch_p)
            # rec_patch = color_dequantize(rec_patch_p,self.clusters)

            # rec_patch = self.conv(rec_patch_p)
            # print(rec_patch_p.shape)
            # dn_input = self.denoise_model_preproc(rec_patch_p)
            # print(dn_input.shape)
            input_patch = rearrange(in_patches[patch],'(n h w) bs d -> bs (n d) h w',h=i_p,w=i_p)             
            rec_patch = self.denoise_model(input_patch)


            # -- dequantize rec --
            # rec_patch_p = rearrange(rec_patch_p,'(n h w) bs d -> bs (n d) h w',h=i_p,w=i_p) 
            # rec_patch = color_dequantize(rec_patch_p,self.clusters,i_p)

            # -- xform inputs are SRC and TGT --
            # st_patch = self.xform(src_patches[patch],tgt_patches[patch])
            # rec_patch_p,_ = self.attn(st_patch,src_patches[patch],in_patches[patch])
            # rec_patch_p = rearrange(st_patch,'(h w) bs d -> bs d h w',h=i_p,w=i_p) 
            # rec_patch_p = rearrange(rec_patch_p,'(h w) bs d -> bs d h w',h=i_p,w=i_p) 
            # rec_patch = self.conv(rec_patch_p)
            # rec_patch_p = rearrange(rec_patch_p,'(h w) bs d -> bs (d h w)',h=i_p,w=i_p) 
            # rec_patch = self.linear(rec_patch_p)
            # rec_patch = rearrange(rec_patch,'bs (c h w) -> bs c h w',h=i_p,w=i_p) 
            # print(rec_patch.shape)

            rec_image.append(rec_patch)
            loss += F.mse_loss(out_patches[patch],rec_patch)

        # rec_0 = torch.clamp(rec_image[0]+0.5,0,1)
        # out_0 = torch.clamp(out_patches[0]+0.5,0,1)
        # tv_utils.save_image(rec_0,'rec_0.png')
        # tv_utils.save_image(out_0,'out_0.png')

        # -- reshape reconstructed patches into an image --
        r = self.im_size // self.output_patch
        if (n_patches == 0):
            rec_image = rec_image[0]
        else:
            rec_image = rearrange(rec_image,'(s1 s2) bs c h w -> bs c (s1 h) (s2 w)',s1=r)

        # -- apply convolution to smooth edge effects --
        # rec_image = self.end_conv(rec_image.detach())

        # rec_img = torch.clamp(rec_image+0.5,0,1)
        # tv_utils.save_image(rec_img,'rec_stitch.png')

        # out_image = rearrange(out_patches,'(s1 s2) bs c h w -> bs c (s1 h) (s2 w)',s1=r)
        # out_img = torch.clamp(out_image+0.5,0,1)
        # tv_utils.save_image(out_img,'out_stitch.png')
        # out_img = torch.clamp(outputs+0.5,0,1)
        # tv_utils.save_image(out_img,'outputs.png')
        # loss += F.mse_loss(outputs,rec_image)

        return loss,rec_image

        
    # def create_patches(self,features_list):
    #     """
    #     features_list (array-like): Batch-Size, Height, Width, Channels
    #     """

    #     features_list
    #     rearrange(pics[0], '(s1 h) (s2 w) c -> (s1 s2) h w c', s1=3, s2=3)
    #     rearrange(src_ftrs,'n (h1 h2) (w1 w2) c -> n h1 w1')
    #     o        
    #     patch_features
    #     for features in features_list:
    #     return 
