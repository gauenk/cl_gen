# -- python imports --
import cv2
import numpy as np
import pandas as pd
import matplotlib as mpl
from pathlib import Path
import numpy.random as npr
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from scipy import stats as sc_stats
from collections import OrderedDict
from einops import rearrange
from skimage import data, color, io, img_as_float

# -- pytorch imports --
import torch
import torchvision.transforms.functional as tvF
import torchvision.utils as tv_utils

# -- faiss imports --
import faiss

# -- project imports --
from pyutils import add_legend,save_image,print_tensor_stats,global_flow_to_blocks,global_blocks_to_pixel,tile_patches
from lpas.main import get_main_config
from datasets.transforms import get_noise_config,get_noise_transform
from explore.wrap_image_data import load_image_dataset,sample_to_cuda

DIR = Path("./output/pretty_plots")

def highlight_patch(patch,color,rx,ry,alpha=0.5):

    H,W,C = patch.shape
    label = np.zeros((H,W,C))
    label[rx,ry,:] = color
    
    annotated = np.copy(patch)
    annotated[rx,ry] = (1 - alpha) * patch[rx,ry] + alpha * label[rx,ry]

    return annotated

def compute_nnf(pair,center,ps,gpuid=0):
    ref = pair[0]
    prop = pair[1]
    rx = slice(center.x - ps//2,center.x + ps//2+1)
    ry = slice(center.y - ps//2,center.y + ps//2+1)
    patch = ref[:,rx,ry]
    C,H,W = ref.shape
    print("patch.shape",patch.shape)
    print("ref.shape",ref.shape)

    # -- define query --
    query = rearrange(patch,'c h w -> 1 (h w c)')
    print("query",query.shape)
    R,ND = query.shape
    # B,N,R,ND = q_patches.shape
    # print(B,N,R,ND,"B,N,R,ND")
    # query_ftr = q_patches.ftr[b,n].contiguous()

    # -- tile proposed patches --
    database,_ = tile_patches(prop[None,None,:],ps)
    database = rearrange(database,'b n r l -> (b n r) l')

    # -- faiss setup --
    res = faiss.StandardGpuResources()
    faiss_cfg = faiss.GpuIndexFlatConfig()
    faiss_cfg.useFloat16 = False
    faiss_cfg.device = gpuid
    
    # -- execute search --
    gpu_index = faiss.GpuIndexFlatL2(res, ND, faiss_cfg)
    gpu_index.add(database)
    K = 10
    D,I = gpu_index.search(query,K)
    print(D)
    print(I)

    # -- get nnf (x,y) from I --
    index = I[0].numpy()
    locs = np.unravel_index(index,(H,W))
    locs = np.c_[locs]
    nnf = []
    for loc in locs:
        nnf_loc = edict()
        nnf_loc.x = loc[0]
        nnf_loc.y = loc[1]
        nnf.append(nnf_loc)
    return nnf

def draw_nnf_arrow(image,pair,ps,FS,pad,nblocks,acolor=(0,0,255)):
    ref_center = edict({'x':FS//2,'y':FS//2})
    nnf = compute_nnf(pair,ref_center,ps)
    for vector in nnf:
        #update_position(vecotr
        apply_pixel_loc_offset(vector,FS,pad)
        image = draw_flow_arrow(image,None,vector,FS,pad,nblocks,acolor)
    return image

def draw_flow_arrow(image,flow,end,FS,pad,nblocks,acolor=(0,0,255)):
    TFS = image.shape[-1]
    center = edict({'x':FS//2+pad,'y':FS//2+pad})
    if end is None:
        end = get_flow_end(flow,center,TFS,FS,pad,nblocks)
    image = draw_arrow(image,center,end,acolor)
    return image

def draw_arrow(image,start,end,acolor=(0,0,255)):
    """
    image.shape = (C,H,W)
    output is 
    image.shape = (C,H,W)


    start.x,start.y
    end.x,end.y
    """
    combo = image.numpy()
    combo = rearrange(combo,'c h w -> h w c')
    combo = combo.copy()
    cv2.arrowedLine(combo, (start.x,start.y),(end.x,end.y), acolor, 1,
                    tipLength=0.1)
    combo = rearrange(torch.FloatTensor(combo),'h w c -> c h w')
    return combo

def get_flow_end(flow,start,TFS,FS,pad,nblocks):

    est_FS = TFS - 3*pad
    assert 2*FS == est_FS, "Target Frame Size v.s. Estimated Frame Size"

    # -- compute frame "ends" --
    end = edict()
    end.x = start.x + flow[0].item()
    end.y = start.y + flow[1].item()

    # -- apply offset --
    apply_pixel_loc_offset(end,FS,pad)
    # OFFSET_IMG2_X = FS + 2*pad
    # OFFSET_IMG2_Y = 0
    # end.x += OFFSET_IMG2_X
    # end.y += OFFSET_IMG2_Y

    return end

def apply_pixel_loc_offset(loc,FS,pad):
    OFFSET_IMG2_X = FS + 2*pad
    OFFSET_IMG2_Y = 0
    loc.x += OFFSET_IMG2_X
    loc.y += OFFSET_IMG2_Y
    return loc

def add_jitter(ndarray,std=0.05):
    return np.random.normal(ndarray,scale=std)

def get_images():
    # -- get experiment config --
    cfg = get_main_config()
    cfg.batch_size = 100
    cfg.nframes = 3
    cfg.frame_size = 350
    cfg.N = cfg.nframes
    cfg.dynamic.frame_size = cfg.frame_size
    cfg.dynamic.frames = cfg.nframes
    cfg.gpuid = 0
    cfg.random_seed = 0    
    T = cfg.nframes

    # -- setup seed --
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    # -- load image data --
    data,loader = load_image_dataset(cfg)
    data_iter = iter(loader.tr)

    # -- get image sample --
    N = 2
    for i in range(N):
        sample = next(data_iter)
    dyn_noisy = sample['noisy'] # dynamics and noise
    dyn_clean = sample['burst'] # dynamics and no noise
    static_noisy = sample['snoisy'] # no dynamics and noise
    static_clean = sample['sburst'] # no dynamics and no noise
    flow = sample['flow']
    # save_image(dyn_clean[T//2],"samples.png")
    pick = 26
    save_image(dyn_clean[T//2,pick],"samples.png")
    cropped = True

    # -- get picks --
    noisy = dyn_noisy[:,pick]+0.5
    clean = dyn_clean[:,pick]+0.5

    # -- optionally crop --
    # T,L = 0,0
    # CS = 175
    T,L = 150,0
    CS = 175
    if cropped:
        noisy = tvF.crop(noisy,T,L,CS,CS)
        clean = tvF.crop(clean,T,L,CS,CS)
    return noisy,clean,flow[pick],cfg.nblocks

def get_fake_denoiser_quality_v_metric_data(metric_name):
    fake = {}
    N = 10+1
    if metric_name == "nnf":
        x = add_jitter(np.arange(N))/N
        y = 36*add_jitter(np.arange(N)[::-1]/N,std=0.1)
        yerr = (np.ones(N))**1.2
        fake['x'] = x
        fake['y'] = y
        fake['yerr'] = yerr
    elif metric_name == "epe":
        x = add_jitter(np.arange(N))/N
        y = [add_jitter(36*npr.rand(1),std=i*0.5) for i in range(N)]
        y = np.r_[y][:,0]
        fake['x'] = x
        fake['y'] = y
    else:
        raise ValueError(f"Uknown metric name [{metric_name}]")
    fake = edict(fake)
    return fake

def plot_denoiser_quality_v_metric():

    # -- gather data --
    fake_nnf = get_fake_denoiser_quality_v_metric_data("nnf")
    fake_epe = get_fake_denoiser_quality_v_metric_data("epe")

    # -- create plot --
    fig,ax = plt.subplots(figsize=(8,4))
    ax.scatter(fake_nnf.x,fake_nnf.y,color='k',ls='None',marker='x')
    # ax.errorbar(fake_nnf.x,fake_nnf.y,fake_nnf.yerr,color='k',ls='-')
    ax.scatter(fake_epe.x,fake_epe.y,color='k',marker='o')
    ax.set_title("Correlation between Metric and Denoiser Quality",fontsize=15)
    ax.set_xlabel("Metric Value",fontsize=15)
    ax.set_ylabel("Denoiser Quality",fontsize=15)

    # -- fit a line --
    nnf_lin = sc_stats.linregress(fake_nnf.x,fake_nnf.y)
    nnf_fit_y = nnf_lin.slope * fake_nnf.x + nnf_lin.intercept
    ax.plot(fake_nnf.x,nnf_fit_y,'k',ls='-')

    of_lin = sc_stats.linregress(fake_epe.x,fake_epe.y)
    of_fit_y = of_lin.slope * fake_epe.x + of_lin.intercept
    ax.plot(fake_epe.x,of_fit_y,'k',ls='--')

    # -- add legend --
    ax = add_legend(ax,'Metric',['NNF','EPE'],framealpha=0.)

    # -- save plot --
    metric_name = "metrics_nnf_and_epe"
    if not DIR.exists(): DIR.mkdir()
    fn =  DIR / f"./denoiser_quality_v_{metric_name}.png"
    plt.savefig(fn,transparent=True,bbox_inches='tight',dpi=300)
    plt.close('all')
    print(f"Wrote plot to [{fn}]")

def plot_examples_nnf_v_epe():

    # -- get images --
    noisy,clean,flow,nblocks = get_images()
    FS = noisy.shape[-1] # frame size
    CFS = 128 # crop frame size
    pics = OrderedDict()
    # pics['Frame t'] = clean[0]
    # pics['Frame t+t'] = clean[1]
    # raw_patches = tvF.resized_crop(clean,95,55,CFS,CFS,(FS,FS))
    print("clean.shape",clean.shape)
    raw_patches = tvF.crop(clean,0,0,CFS,CFS)
    # raw_patches = tvF.crop(clean,95,55,CFS,CFS)
    T,C,H,W = raw_patches.shape
    # patches = torch.zeros((T,C+1,H,W))
    # patches[:,:-1,:,:] = raw_patches.clone()
    patches = rearrange(raw_patches,'t c h w -> t h w c')
    alpha = 0.6
    print("patches.shape",patches.shape)

    # -- color indices for ref index -- 
    patch = patches[0].clone().numpy()
    patch_0 = patches[0].clone()
    patch_1 = patches[1].clone()
    # ref_shade = np.zeros((H,W,3))
    # print(ref_shade.shape)
    # rx = slice(CFS//2,CFS//2+10)
    # ry = slice(CFS//2,CFS//2+10)
    # ref_shade[rx,ry,:] = [1,0,0]
    # #ref_shade[CFS//2:CFS//2+10,CFS//2:CFS//2+10,:] = [1,0,0]
    # ave = patches[0].clone().numpy()
    # ave[rx,ry] = (1 - alpha) * patch[rx,ry] + alpha * ref_shade[rx,ry]

    rx = slice(CFS//2,CFS//2+10)
    ry = slice(CFS//2,CFS//2+10)
    acolor = [0,1,1]
    print("patch.shape",patch.shape)
    annotated = highlight_patch(patch,acolor,rx,ry,alpha=0.5)
    print("annotated.shape",annotated.shape)
    annotated *= 255.
    annotated = np.float32(annotated.astype(np.uint8)).copy()
    print(annotated.shape)
    print(type(annotated))
    print(annotated.dtype)
    # annotated = np.zeros((512,512,3),np.uint8)
    print(patch_0.shape)
    stack = rearrange(torch.stack([patch_0,patch_1],dim=0),'b h w c -> b c h w')
    print("stack.shape",stack.shape)
    combo = tv_utils.make_grid(stack,nrow=2,padding=2,pad_value=0)
    print(combo.shape)
    # combo = rearrange(combo,'c h w -> h w c')*255.
    print_tensor_stats("combo",combo)
    print("combo.shape",combo.shape)

    pad = 2
    ps = 3
    pair = stack
    combo = draw_nnf_arrow(combo,pair,ps,CFS,pad,nblocks,acolor=(0,255,0))
    combo = draw_flow_arrow(combo,flow[0],None,CFS,pad,nblocks,acolor=(0,0,255))

    # combo = combo.numpy()
    # combo = rearrange(combo,'c h w -> h w c')
    # print(combo.shape)
    # combo = combo.copy()

    # mid = edict()
    # mid.x = FS // 2
    # mid.y = FS // 2
    # tgt = edict()
    # IMG2_X = W + 2
    # IMG2_Y = 0
    # tgt.x = FS //2 + IMG2_X
    # tgt.y = FS //2 + IMG2_Y
    # print(mid)
    # print(tgt)
    # cv2.arrowedLine(combo, (mid.x,mid.y),(tgt.x,tgt.y), (0,0,255), 2,
    #                 tipLength=0.2)
    # combo = rearrange(combo,'h w c -> c h w')

    patch = rearrange(torch.FloatTensor(patch),'h w c -> c h w')
    # pics['Patch t'] = patch
    annotated = rearrange(torch.FloatTensor(annotated),'h w c -> c h w')
    pics['combo'] = torch.FloatTensor(combo)


    # patch = color.rgb2hsv(patch)
    # ref_shade = color.rgb2hsv(ref_shade)
    # print(ref_shade.shape)
    # patch[...,0] = ref_shade[...,0]
    # # patch[...,1] = ref_shade[...,1] * alpha# + (1 - alpha) * patch[...,1]
    # patch_shaded = color.hsv2rgb(patch)
    # pics['Patch t'] = rearrange(torch.FloatTensor(patch_shaded),'h w c -> c h w')
    # pics['Patch t+1'] = rearrange(torch.FloatTensor(patch_shaded),'h w c -> c h w')

    # patch_shaded = color.label2rgb(ref_shade,patch,alpha=.3,bg_label=0,kind='overlay',saturation=0.6)
    # print("patch_shaded.shape",patch_shaded.shape)
    # pics['Patch t'] = rearrange(torch.FloatTensor(patch_shaded),'h w c -> c h w')
    # pics['Patch t+1'] = rearrange(torch.FloatTensor(patch_shaded),'h w c -> c h w')

    # -- color indices for optical flow -- 
    print(flow.shape)
    blocks = global_flow_to_blocks(flow[None,:],nblocks)
    print("blocks",blocks)
    print(blocks.shape)
    of_label = patches[1].clone()
    print(of_label.shape)


    # -- ALL PICS --
    names = list(pics.keys())
    M = len(names)
    dpi = 300
    #dpi = mpl.rcParams['figure.dpi']
    H,W = clean.shape[-2:]
    print(H,W)
    buf = 10
    mod = 3*buf
    figsize = M * (1.5*W / float(dpi)), (H-mod) / float(dpi)
    print(figsize)
    fig,axes = plt.subplots(1,M,figsize=figsize,sharex=True,sharey=True)
    if not isinstance(axes,list): axes = [axes]
    else: fig.subplots_adjust(hspace=0.05,wspace=0.05)
    for i,ax in enumerate(axes):
        name = names[i]
        pic = rearrange(pics[name],'c h w -> h w c')
        pic = torch.clip(pic,0,1)
        S = pic.shape[1]
        print(pic.shape)
        pic = tvF.crop(pic.transpose(2,0),buf,0,S,S-mod)
        pic = pic.transpose(2,0)
        print(pic.shape)
        ax.imshow(pic,interpolation='none',aspect='auto')
        ax.set_ylabel('')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # label = textwrap.fill(name + '\n' + '1.2',15)
        label = name
        # ax.set_xlabel(label,fontsize=12)

    # -- plot function --
    if not DIR.exists(): DIR.mkdir()
    fn =  DIR / "./examples_nnf_v_epe.png"
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0) 
    plt.savefig(fn,transparent=True,dpi=300)
    #plt.savefig(fn,transparent=True,bbox_inches='tight',dpi=300,pad_inches=0)
    plt.close('all')
    print(f"Wrote plot to [{fn}]")

def run():

    plot_denoiser_quality_v_metric()
    plot_examples_nnf_v_epe()

