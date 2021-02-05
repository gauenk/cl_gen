
# -- python imports --
import pdb
import numpy as np
import numpy.random as npr
from einops import rearrange, repeat, reduce
from easydict import EasyDict as edict
import matplotlib.pyplot as plt

# -- pytorch imports --
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils as tv_utils

# -- project imports --
import settings
from layers.simcl import ClBlockLoss
from layers.stn import STN_Net

# -- [local] project imports --
from .utils import color_quantize,color_dequantize,plot_patches

# -- 3d party imports --
from performer_pytorch import Performer,PerformerLM,FastAttention

class TransformerNetwork32_dip_v2(nn.Module):

    def __init__(self, ftr_model, d_model, im_size, input_patch, output_patch, n_frames, bw=False, denoise_model = None, batch_size = 8):
        super(TransformerNetwork32_dip_v2, self).__init__()
        """
        Transformer for denoising

        im_size : int : the length and width of the square input image

        """
        self.d_model = d_model
        self.ftr_model = {'spoof':ftr_model}
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
        dropout = 0.0
        xform_args = [d_model, nhead, num_enc_layers, num_dec_layers, dim_ff, dropout]

        self.color_code = nn.Linear(d_model,3)

        d_model = 3
        # self.perform = Performer(d_model,8,1) # 8,4
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
        self.attn = nn.MultiheadAttention(3,nhead,dropout,qkv_same_params=False)#,vdim=vdim)
        # self.stn = STN_Net()

        # -- keep output to be shifted inputs --
        # print(dir(self.attn))
        # print(self.attn.out_proj)
        # print(self.attn.v_proj_weight)
        self.attn.v_proj_weight.data = torch.eye(3)
        # # print(self.attn.v_proj_weight)
        self.attn.v_proj_weight = self.attn.v_proj_weight.requires_grad_(False)
        # # print('o',self.attn.out_proj.weight)
        self.attn.out_proj.weight.data = torch.eye(3)
        # # print('o',self.attn.out_proj.weight.data)
        # # print('b',self.attn.out_proj.bias.data)
        self.attn.out_proj.bias.data = torch.zeros_like(self.attn.out_proj.bias.data)
        # # print('b',self.attn.out_proj.bias.data)
        self.attn.out_proj = self.attn.out_proj.requires_grad_(False)
        

        # self.attn = nn.MultiheadAttention(d_model,nhead,dropout)#,vdim=vdim)
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
        # self.conv = nn.Sequential(*[nn.Conv2d(in_channels,out_chann,3,stride,padding)])

        # -- using MEAN as denoiser --
        in_channels = d_model
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels,out_chann,3,stride,padding)])

        self.end_conv = nn.Sequential(*[nn.Conv2d(3,3,1),nn.LeakyReLU(),nn.Conv2d(3,3,1)])

        padding = (input_patch - output_patch) // 2
        # stride = im_size-input_patch
        stride = output_patch
        # print(input_patch,padding,stride)
        self.unfold_input = nn.Unfold(input_patch,1,padding,stride)
        # self.linear = nn.Linear(ftr_size,img_size)

    def forward(self, inputs, outputs, step, max_steps, loss_diff):
        """
        inputs (array-like): (BS,N,H,W,C)
        outputs (array-like): (BS,N,H,W,C)

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
        # if (N % 2) == 1:
        #     middle_index = N//2
        #     inputs[:,middle_index] = torch.normal(inputs[:,middle_index],std=self.std)

        i_p,o_p = self.input_patch,self.output_patch
        # m_index = N//2
        # no_mid = np.r_[np.arange(m_index),np.arange(m_index+1,N)]


        # -- feature extraction --
        fm_inputs = rearrange(inputs,'bs n c h w -> n bs c h w')
        if C == 1: fm_inputs = repeat(fm_inputs,'n bs 1 h w -> n bs c h w',c=3)
        # src_ftrs = self.ftr_model['spoof'](fm_inputs).detach()
        # tgt_ftrs = self._ftr_model(rearrange(outputs,'bs c h w -> 1 bs c h w')).detach()[0]

        # print(fm_inputs.shape)
        # src_ftrs = fm_inputs
        # -- each pixel value turns into indicator var --
        src_ftrs = color_quantize(fm_inputs,self.clusters,self.d_model)
        # qoutputs = color_quantize(outputs,self.clusters,self.d_model)
        # print(src_ftrs.shape)

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
        # print('1) --> a',inputs.shape)
        # print(rearrange(inputs,'bs n c h w -> (bs n) c h w').shape)
        in_patches = self.unfold_input(rearrange(inputs,'bs n c h w -> (bs n) c h w'))
        # print('2) --> b',in_patches.shape)
        in_patches = rearrange(in_patches,'(b n) (c i) r -> r n i b c',b=BS,i=i_p**2)
        in_patches_flat = rearrange(in_patches,'r n i b c -> r (n i) b c',b=BS,i=i_p**2)
        # in_patches_mid = rearrange(inputs[:,N//2],'(b n) (c i) r -> r (n i) b c',b=BS,i=i_p**2)
        in_patches_mid = rearrange(self.unfold_input(inputs[:,N//2]),'b (c i) r -> r i b c',i=i_p**2)
        

        # print("ip.shape",in_patches.shape)
        # plot_patches(in_patches,N,i_p)
        # in_patches = rearrange(inputs,'bs n c (rh h) (rw w) -> (rh rw) (n h w) bs c',h=i_p,w=i_p)
        # out_patches = rearrange(outputs,'bs c (rh h) (rw w) -> (rh rw) bs c h w',h=o_p,w=o_p)
        out_patches = self.unfold_input(rearrange(outputs,'bs n c h w -> (bs n) c h w'))
        # print("op.shape",out_patches.shape)
        out_patches = rearrange(out_patches,'(b n) (c i) r -> r (n i) b c',b=BS,i=i_p**2)
        # print("op.shape",out_patches.shape)
        # print('--> a',src_ftrs.shape)
        # print('**-->',rearrange(src_ftrs,'n bs c h w -> (n bs) c h w').shape)
        # print(rearrange(src_ftrs,'n bs c h w -> (n bs) c h w')[0][0][0][0])
        src_patches = self.unfold_input(rearrange(src_ftrs,'n bs c h w -> (n bs) c h w'))
        # print('--> b',src_patches.shape)
        # print(src_patches[0][0][0])
        src_patches = rearrange(src_patches,'(n b) (c i) r -> r (n i) b c',b=BS,i=i_p**2)
        src_patches_mid = rearrange(self.unfold_input(src_ftrs[N//2,:]),'b (c i) r -> r i b c',i=i_p**2)

        # print('--> c',src_patches.shape)
        # src_patches=rearrange(src_ftrs,'n bs c (rh h) (rw w) -> (rh rw) (n h w) bs c',h=i_p,w=i_p)
        # tgt_patches = rearrange(tgt_ftrs,'bs c (rh h) (rw w) -> (rh rw) (h w) bs c',h=i_p,w=i_p)
        assert in_patches.shape[0] == src_patches.shape[0], "Number of patches must match"
        assert in_patches.shape[0] == out_patches.shape[0], "Number of patches must match"
        n_patches = in_patches.shape[0]

        # -- compute transformer for each patch --; [todo] Patches concats with BS
        rec_image = []
        loss = 0
        # print("n_patches",n_patches)
        for patch in range(n_patches):

            # Why does each rec image init as a blob? We want it to be *almost* a copy of the input image.
            # with attention, you cannot just copy by index. maybe an adv over CNN, which might be able to do this
            # -- start with attn directly on pixels --
            attn_in_patch,attn_map = [],[]
            for i in range(N):
                # a = self.stn(in_patches[patch][i].reshape(1,3,32,32))
                # a = a[0].reshape(32*32,1,3)
                a,b = self.attn(in_patches[patch][i],in_patches_flat[patch],in_patches_flat[patch])
                attn_in_patch.append(a)
                #attn_map.append(b)
            attn_in_patch = torch.cat(attn_in_patch,dim=0)
            # print(attn_in_patch.shape)
            # attn_in_patch,w_attn = self.attn(in_patches[patch],in_patches_flat[patch],in_patches_flat[patch])
            # repeats = 3
            # for r in range(repeats):
            #     attn_in_patch,w_attn = self.attn(attn_in_patch,attn_in_patch,attn_in_patch)
            # w_attn = w_attn[0]
            
            # -- regularize against large jumps --
            # print('ratio',torch.mean(torch.diag(w_attn,0)).item()/torch.mean(w_attn).item())
            # loss +=  10**8 * torch.norm( w_attn - torch.eye(w_attn[0].shape[0]).cuda())
            # loss -=  10**10 * torch.sum(torch.abs( torch.diag(w_attn,0) ))
            # print("0",torch.sum(w_attn,dim=0))
            # print("1",torch.sum(w_attn,dim=1))
            # print("2",torch.sum(w_attn,dim=2))
            # print("aip",attn_in_patch.shape)

            st_patch = attn_in_patch
            # st_patch = self.perform(attn_in_patch)

            # -- xform inputs are SRC and SRC --
            # print(src_patches[patch].shape)
            # dn_src_in = rearrange(src_patches[patch],'(n i_h i_w) b c -> b (n c) i_h i_w',n=N,i_h=i_p,i_w=i_p)
            # # print('a',dn_src_in.shape)
            # dn_src_out = self.denoise_model_preproc_a(dn_src_in)
            # # print('b',dn_src_out.shape)
            # dn_src_out = rearrange(dn_src_out,'b (n c) i_h i_w -> (n i_h i_w) b c',n=N,i_h=i_p,i_w=i_p)
            # # print('c',dn_src_out.shape)
            # st_patch = self.perform(dn_src_out)

            # -- d_attn to color channels --
            # st_patch_l = rearrange(st_patch,'l b c -> (l b) c')
            # st_patch_c3 = self.color_code(st_patch_l)
            # st_patch_c3 = rearrange(st_patch_c3,'(l b) c -> l b c',b=BS)
            # st_patch_c3 = st_patch

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
            # st_patch,_ = self.attn(st_patch,src_patches[patch],in_patches[patch])
            # st_patch,_ = self.attn(st_patch,src_patches[patch],src_patches[patch])
            # st_patch,w_attn = self.attn(st_patch_c3,in_patches[patch],in_patches[patch])

            # w_attn = w_attn[0]
            # print('o',self.attn.out_proj.weight)
            # print('v',self.attn.v_proj_weight)
            # print("w",w_attn.shape)

            # -- plot attn map --

            # randint = np.random.randint(100)
            # for idx in range(N):
            #     plt.imshow(attn_map[i][0].detach().cpu())
            #     save_fn = f"w_attn_{randint}_{idx}.png"
            #     print("Att Map",save_fn)
            #     plt.savefig(save_fn,dpi=500)
            #     plt.close("all")
            
            # -- plot attn map v2 --
            # w_attn = rearrange(w_attn,'(n_1 i_h_1 i_w_1) (n_2 i_h_2 i_w_2) -> (n_1 n_2) (i_h_1 i_h_2) (i_w_1 i_w_2)',n_1=N,n_2=N,i_h_1=i_p,i_h_2=i_p,i_w_1=i_p,i_w_2=i_p)
            # w_attn = rearrange(w_attn,'(n_1 i_1) (n_2 i_2) -> (n_1 n_2) i_1 i_2',n_1=N,n_2=N)
            # w_attn = w_attn.unsqueeze(1).detach().cpu()
            # print(w_attn[0][:10][:10],w_attn.std(),w_attn.mean(),w_attn.min(),w_attn.max())
            # save_block = tv_utils.make_grid(w_attn,normalize=True,nrow=3,padding=10,pad_value=0).transpose(0,2)
            # plt.imshow(save_block)
            # plt.savefig("w_attn_block.png",dpi=500)
            # plt.close("all")

            # print(st_patch.shape,in_patches[patch].shape)
            # rec_patch_p,_ = self.attn(st_patch,st_patch,in_patches[patch])
            # rec_patch_p,_ = self.attn(st_patch,st_patch,nm_patches[patch])
            # rec_patch_p,_ = self.attn(nm_patches[patch],st_patch,st_patch)
            # rec_patch_p = st_patch


            # -- averaging scheme --
            # in_patches[patch] :: (N x I) x B x C
            # st_patches :: (N x I) x B x C
            # st_patch_ni = rearrange(st_patch,'(n i) b c -> n i b c',n=N) 
            # print(src_patches[patch].shape)
            in_patches_mid_r = in_patches_mid[patch].unsqueeze(0).repeat(N,1,1,1)
            # src_patches_mid_r = src_patches_mid[patch].unsqueeze(0).repeat(N,1,1,1)
            # print("ave scheme",st_patch.shape,in_patches_mid[patch].shape)
            # print(st_patch_ni.shape,src_patches_mid_r.shape)
            # loss += F.mse_loss(st_patch_ni,src_patches_mid_r)
            # loss += F.mse_loss(st_patch_ni,in_patches_mid_r)
            
            # -- conv to rec size --
            # rec_patch_p = rearrange(st_patch,'(n h w) bs d -> bs (n d) h w',h=i_p,w=i_p) 
            # rec_patch_p = rearrange(rec_patch_p,'(n h w) bs d -> bs (n d) h w',h=i_p,w=i_p) 
            rec_patch_p = rearrange(st_patch,'(n h w) bs d -> (bs n) d h w',h=i_p,w=i_p) 
            
            # -- create aligned imgs -- 
            align_imgs = rec_patch_p
            # align_imgs = self.denoise_model_preproc_b(rec_patch_p)
            # print("rec/align",rec_patch_p.shape,align_imgs.shape)

            # -- use constrastive loss --
            # align_imgs = rearrange(align_imgs,'(bs n) d h w -> n bs d h w',bs=BS)
            # cl_loss = 0
            # if self.training: cl_loss = self.simclr_loss(align_imgs) 
            # loss += cl_loss

            # -- add extra noise to middle output patch --
            # nin_patch = torch.normal(out_patches[patch].repeat(N,1,1,1),std=self.std)
            # nin_patch = rearrange(nin_patch,'(bs n) a b c -> (n bs) a b c',bs=BS)

            # nin_patch = torch.normal(out_patches[patch].repeat(1,N,1,1),std=self.std)
            # nin_patch = torch.normal(out_patches[patch],std=self.std)
            # nin_patch = rearrange(nin_patch,'(bs n) a b c -> n bs a b c',bs=BS)

            nin_patch = rearrange(out_patches[patch],'(n h w) 1 c -> n c h w',h=i_p,w=i_p)
            # print(nin_patch.shape)
            # print(out_patches[patch].shape,align_imgs.shape,nin_patch.shape)

            # -- alignment loss across each image -- 
            # print(nin_patch.shape,align_imgs.shape)
            nin_patch_mid = nin_patch[N//2].repeat(N,1,1,1).contiguous()
            align_imgs = align_imgs.contiguous()
            # print(nin_patch_mid.shape,align_imgs.shape,nin_patch.shape)
            mid_mse_loss = F.mse_loss(nin_patch_mid,align_imgs)
            loss += mid_mse_loss
            mid_mse_loss_item = mid_mse_loss.item()

            all_mse_loss = F.mse_loss(nin_patch,align_imgs)
            all_mse_loss_item = all_mse_loss.item()
            # print("m",mid_mse_loss / all_mse_loss_item)
            # loss += mid_mse_loss/all_mse_loss_item
            # loss_diff = torch.clamp(torch.Tensor([loss_diff]),0,1)[0]
            # -- to help it "get started" --
            loss += loss_diff * (1 - step/max_steps)**10 * all_mse_loss 
            # loss += (1 - step/max_steps)**2 * all_mse_loss
            # loss += all_mse_loss
            
            # print(mid_mse_loss_item,all_mse_loss_item)
            # if mid_mse_loss < all_mse_loss_item:
            #     loss += all_mse_loss
            # else:
            #     loss += (mid_mse_loss/all_mse_loss_item)**2 * all_mse_loss                

            # print(loss.item())
            # loss += F.mse_loss(nin_patch[1],align_imgs)
            # rec_patch_p = rec_patch_p.detach() # cut off gradient for denoising here

            # -- use MEAN as the denoising layer --
            # print(st_patch.shape)
            # rec_patch_p = rearrange(st_patch.detach(),'(n h w) bs d -> bs n d h w',h=i_p,w=i_p) 
            # print(rec_patch_p.shape)
            # rec_patch_p = torch.mean(rec_patch_p,dim=1)
            # print(rec_patch_p.shape)
            # rec_patch = self.conv(rec_patch_p)
            # print("rec",rec_patch.shape)

            # -- V2: use mean as denoiser --
            rec_patch = torch.mean(align_imgs,dim=0).contiguous()
            # rec_patch = align_imgs[N//2]
            # mid_noise = align_imgs[N//2] - in_patches[patch][N//2].reshape(32,32,3).transpose(0,2)
            # not_middle = set(range(N)) - set([N//2])
            # print(not_middle)
            # mnoise_imgs = torch.cat([align_imgs[i] + mid_noise for i in not_middle])
            # for j in range(N):
            #     noise = align_imgs[N//2] - rec_patch
            #     for i in range(N):
            #         if i == j: continue
            # for i in not_middle:
            #     # print(mid_noise.shape,align_imgs[i].shape)
            #     new_noise_img = align_imgs[i] + mid_noise
            #     # print(new_noise_img.shape)
            #     old_noise_img = in_patches[patch][i].reshape(32,32,3).transpose(0,2)
            #     # print(old_noise_img.shape)
            #     loss += F.mse_loss(new_noise_img,old_noise_img)


            # rec_patch = align_imgs[N//2].contiguous()

            # print("rec_v2",rec_patch.shape)

            # -- use reconstructed patches form above for final denoising layer --
            # rec_patch_p = rearrange(st_patch.detach(),'(n h w) bs d -> bs (n d) h w',h=i_p,w=i_p) 

            # rec_patch_p = rearrange(st_patch.detach(),'(bs n) d h w -> (n h w) bs d',bs=BS)

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

            # nin_patch_mid = nin_patch[N//2].contiguous()
            
            # print("Saving Figures")
            # r_save = rec_patch.detach().cpu().unsqueeze(0)
            # a_save = align_imgs.detach().cpu()
            # # print(r_save.shape,align_imgs.shape)
            # # save_img = torch.cat([r_save,a_save],dim=0)
            # # print(save_img.shape)

            # inp = rearrange(in_patches_flat[patch],'(n i_h i_w) 1 c -> n c i_h i_w',i_h=i_p,i_w=i_p).detach().cpu()
            # oup = nin_patch_mid.detach().cpu().unsqueeze(0)
            # inp = inputs[0].cpu()
            # print(inp.shape,oup.shape)
            # save_img_ = torch.cat([inp,oup],dim=0)
            # print(save_img_.shape)

            # save_img = torch.cat([inp,oup,r_save,a_save],dim=0)
            # print("i",inp.mean(),inp.min(),inp.max())
            # print("r",r_save.mean(),r_save.min(),r_save.max())
            # print("a",a_save.mean(),a_save.min(),a_save.max())
            # print("o",oup.mean(),oup.min(),oup.max())
            # print("s",save_img.mean(),save_img.min(),save_img.max())
            # save_img += 0.5
            # save_img = save_img.clamp(0,1)
            # print(save_img.mean(),save_img.min(),save_img.max())
            # print(save_img.shape)
            # save_block = tv_utils.make_grid(save_img,normalize=True,nrow=4).transpose(0,2)
            # plt.imshow(save_block)
            # save_fn = "rec_patch_{:.2f}.png".format(np.random.rand())
            # print(f"Writing image {save_fn}")
            # plt.savefig(save_fn,dpi=300)
            # plt.close("all")


            rec_image.append(rec_patch.unsqueeze(0))
            # print(out_patches.shape,out_patches[patch].shape)
            # print(nin_patch.shape,rec_patch.shape)
            # print(nin_patch[1].shape,rec_patch.shape)
            # loss += F.mse_loss(out_patches[patch],rec_patch)
            # print(nin_patch[N//2].shape,rec_patch.shape)
            # nin_patch_mid = nin_patch[N//2].contiguous()
            # loss += F.mse_loss(nin_patch_mid,rec_patch)
            # print(loss.item())

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

        rec_image = rec_image.contiguous()
        # -- apply convolution to smooth edge effects --
        # rec_image = self.end_conv(rec_image.detach())

        # rec_img = torch.clamp(rec_image+0.5,0,1)
        # tv_utils.save_image(rec_img,'rec_stitch.png')

        # out_image = rearrange(out_patches,'(s1 s2) bs c h w -> bs c (s1 h) (s2 w)',s1=r)
        # out_img = torch.clamp(out_image+0.5,0,1)
        # tv_utils.save_image(out_img,'out_stitch.png')
        # out_img = torch.clamp(outputs+0.5,0,1)
        # tv_utils.save_image(out_img,'outputs.png')
        # print("o",outputs.shape,rec_image.shape)
        rec_loss = F.mse_loss(outputs[:,N//2],rec_image)
        rec_loss_item = rec_loss.item()
        loss += rec_loss
        # print(rec_loss_item)

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
