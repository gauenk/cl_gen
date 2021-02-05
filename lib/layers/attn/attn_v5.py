
# -- python imports --
import pdb
import numpy as np
import numpy.random as npr
from einops import rearrange, repeat, reduce

# -- pytorch imports --
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils as tv_utils

# -- project imports --
import settings

# -- local project imports --
from .utils import color_quantize,color_dequantize
from .position_encoder import PositionalEncoder

# -- 3d party imports --
from performer_pytorch import Performer,PerformerLM,FastAttention

class TransformerNetwork32_v5(nn.Module):

    def __init__(self, ftr_model, d_model, im_size, input_patch, output_patch, n_frames, bw=False, unet = None):
        super(TransformerNetwork32_v5, self).__init__()
        """
        Transformer for denoising

        im_size : int : the length and width of the square input image

        """
        self.d_model = d_model
        self.ftr_model = ftr_model
        # self.denoise_model_preproc_a = nn.Conv2d(d_model*n_frames,d_model*n_frames,3,1,1)
        # self.denoise_model_preproc = nn.Conv2d(d_model*n_frames,3*n_frames,3,1,1)
        # self.denoise_model_preproc_b = nn.Conv2d(d_model*n_frames,3,3,1,1)
        self.unet = unet
        self.std = 5./255
        nhead = 4
        num_enc_layers = 6
        num_dec_layers = 6
        dim_ff = 512
        dropout = 0.1
        # xform_args = [d_model, nhead, num_enc_layers, num_dec_layers, dim_ff, dropout]
        d_model = 3*3
        self.performer = Performer(d_model,16,3)
        # self.xform = nn.Transformer(*xform_args)
        # self.clusters = np.load(f"{settings.ROOT_PATH}/data/kmeans_centers.npy")

        # kwargs = {'num_tokens':512,'max_seq_len':input_patch**2,
        #           'dim':512, 'depth':6,'heads':4,'causal':False,'cross_attend':False}
        # self.xform_enc = PerformerLM(**kwargs)
        # self.xform_dec = PerformerLM(**kwargs)

        nhead = 1
        vdim = 1 if bw else 3 * n_frames
        self.pos_enc = PositionalEncoder(3*n_frames,32*32)
        self.attn = nn.MultiheadAttention(d_model*n_frames,nhead,dropout,vdim=d_model,qkv_same_params=False)
        self.input_patch,self.output_patch = input_patch,output_patch
        self.im_size = im_size

        # -- fix id to output --
        self.attn.v_proj_weight.data = torch.eye(9)
        self.attn.v_proj_weight = self.attn.v_proj_weight.requires_grad_(False)
        self.attn.out_proj.weight.data = torch.eye(9)
        self.attn.out_proj.bias.data = torch.zeros_like(self.attn.out_proj.bias.data)
        self.attn.out_proj = self.attn.out_proj.requires_grad_(False)


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
        # self.conv = nn.Sequential(*[nn.Conv2d(in_channels,out_chann,3,stride,padding)])

        # self.end_conv = nn.Sequential(*[nn.Conv2d(3,3,1),nn.LeakyReLU(),nn.Conv2d(3,3,1)])

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

        # -- including a middle one; we use odd N in this case --
        if (N % 2) == 1:
            middle_index = N//2
            inputs[:,middle_index] = torch.normal(inputs[:,middle_index],std=self.std)

        i_p,o_p = self.input_patch,self.output_patch
        # m_index = N//2
        # no_mid = np.r_[np.arange(m_index),np.arange(m_index+1,N)]


        # -- feature extraction --
        fm_inputs = rearrange(inputs,'bs n c h w -> n bs c h w')
        if C == 1: fm_inputs = repeat(fm_inputs,'n bs 1 h w -> n bs c h w',c=3)
        # src_ftrs = self.ftr_model(fm_inputs).detach()
        # tgt_ftrs = self.ftr_model(rearrange(outputs,'bs c h w -> 1 bs c h w')).detach()[0]

        # -- each pixel value turns into indicator var --
        src_ftrs = fm_inputs
        # src_ftrs = color_quantize(fm_inputs,self.clusters,self.d_model)
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

        # -- noisy targets for alignment step --
        # noisy_inputs = torch.normal(inputs,self.std)
        # rand_order = npr.permutation(N)
        # noisy_inputs[:,:] = noisy_inputs[:,rand_order]
        # nin_patches = self.unfold_input(rearrange(noisy_inputs,'bs n c h w -> (bs n) c h w'))
        # nin_patches = rearrange(nin_patches,'(b n) (c ih iw) r -> r b (n c) ih iw',b=BS,ih=i_p,iw=i_p)

        # -- standard inputs/outputs --
        in_patches = rearrange(inputs,'bs n c h w -> (bs n) c h w')
        in_patches = self.unfold_input(in_patches)
        in_patches = rearrange(in_patches,'(b n) (c i) r -> r n i b c',b=BS,i=i_p**2)
        in_patches_flat = rearrange(in_patches,'r n i b c -> r i b (n c)',b=BS,i=i_p**2)
        in_patches_mid = rearrange(self.unfold_input(inputs[:,N//2]),'b (c i) r -> r i b c',i=i_p**2)


        # in_patches = rearrange(in_patches,'(b n) (c i) r -> r (n i) b c',b=BS,i=i_p**2)
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
            # dn_src_in = rearrange(src_patches[patch],'(n i_h i_w) b c -> b (n c) i_h i_w',n=N,i_h=i_p,i_w=i_p)
            # dn_src_out = self.denoise_model_preproc_a(dn_src_in)
            # dn_src_out = rearrange(dn_src_out,'b (n c) i_h i_w -> (n i_h i_w) b c',n=N,i_h=i_p,i_w=i_p)
            # st_patch = self.perform(dn_src_out)

            # print(in_patches_flat[patch].shape)
            ifp_enc = self.pos_enc(in_patches_flat[patch])
            # print(ifp_enc.shape,in_patches_flat[patch].shape)
            attn_patch = self.performer(ifp_enc)#,ifp_enc,in_patches_flat[patch])
            # attn_patch,_ = self.attn(ifp_enc,ifp_enc,ifp_enc)
            # print(attn_patch.shape)
            rec_patch  = rearrange(attn_patch,'(h w) bs (n c) -> bs (n c) h w',h=i_p,w=i_p,n=N) 
            rec_patch = self.unet(rec_patch) # N -> N-1
            rec_patch  = rearrange(rec_patch,'bs (n c) h w -> bs n c h w',h=i_p,w=i_p,n=N-1) 
            rec_patch_b = rec_patch
            rec_patch_bn  = rearrange(rec_patch,'bs n c h w -> (bs n) c h w')
            # print("r",rec_patch.shape)


            # -- equiv statistics loss --
            r_middle_img = out_patches[patch].repeat(N-1,1,1,1)
            # print(r_middle_img.shape,rec_patch_bn.shape,out_patches[patch].shape)
            mean_est = torch.mean(r_middle_img - rec_patch_bn, dim=(1,2,3))
            mse_loss = F.mse_loss(r_middle_img,rec_patch_bn,reduction='none')
            std_est = torch.flatten(torch.mean( mse_loss, dim=(1,2,3) ))
            # dist_loss = torch.norm(std_est.unsqueeze(1) - std_est)
            dist_loss = 0
            for i in range(N-1):
                for j in range(N-1):                
                    if i >= j: continue
                    si,sj = std_est[i],std_est[j]
                    dist_loss += torch.abs(mean_est[i] - mean_est[j])
                    dist_loss += si + sj - 2 * (si * sj)**0.5

            # -- reconstruction loss --
            rec_patch = torch.mean(rec_patch,dim=1)
            mse_loss = F.mse_loss(out_patches[patch],rec_patch)

            loss += mse_loss + 0.1*dist_loss/(1+mse_loss)
            rec_image.append(rec_patch)




            # st_patch = self.perform(src_patches[patch])

            # st_patch = src_patches[patch]
            # st_patch = self.xform(src_patches[patch],src_patches[patch])
            # st_patch = self.xform(st_patch,nm_patches[patch])
            # print(src_patches[patch].shape)
            # src_patches[patch] = src_patches[patch].type(torch.long)
            # st_patch = self.xform_enc(src_patches[patch][0].type(torch.long), return_encodings = True)
            # st_patch = self.xform_dec(in_patches[patch], context = st_patch)

            # -- final attention to "go back to pixel space"; don't need it for quant --
            # print(st_patch.shape)
            
            # rec_patch_p,_ = self.attn(src_patches[patch],src_patches[patch],in_patches[patch])
            # rec_patch_p,_ = self.attn(st_patch,src_patches[patch],in_patches[patch])
            # rec_patch_p,_ = self.attn(st_patch,st_patch,in_patches[patch])
            # pdb.set_trace()
            # rec_patch_p,_ = self.attn(st_patch,st_patch,nm_patches[patch])
            # rec_patch_p,_ = self.attn(nm_patches[patch],st_patch,st_patch)
            # rec_patch_p = st_patch

            # -- conv to rec size --
            # rec_patch_p = rearrange(st_patch,'(n h w) bs d -> bs (n d) h w',h=i_p,w=i_p) 
            # rec_patch_p = rearrange(rec_patch_p,'(n h w) bs d -> bs (n d) h w',h=i_p,w=i_p) 
            
            # -- create aligned imgs -- 
            # align_imgs = self.denoise_model_preproc_b(rec_patch_p)

            # -- add extra noise to middle output patch --
            # nin_patch = torch.normal(out_patches[patch].repeat(1,N,1,1),std=self.std)
            # nin_patch = torch.normal(out_patches[patch],std=self.std)

            # -- alignment loss -- 
            # loss += F.mse_loss(nin_patch,align_imgs)
            # rec_patch_p = rec_patch_p.detach() # cut off gradient for denoising here

            # rec_patch = self.conv(rec_patch_p)
            # rec_patch = color_dequantize(rec_patch_p,self.clusters)

            # rec_patch = self.conv(rec_patch_p)
            # print(rec_patch_p.shape)

            # print(dn_input.shape)
            # rec_patch = self.denoise_model(rec_patch_p)

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

        # rec_0 = torch.clamp(rec_image[0]+0.5,0,1)
        # out_0 = torch.clamp(out_patches[0]+0.5,0,1)
        # tv_utils.save_image(rec_0,'rec_0.png')
        # tv_utils.save_image(out_0,'out_0.png')

        # -- reshape reconstructed patches into an image --
        r = self.im_size // self.output_patch
        if (n_patches == 0):
            rec_image = rec_image[0]
        else:
            # print(len(rec_image),rec_image[0].shape,r,self.im_size)
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
    
