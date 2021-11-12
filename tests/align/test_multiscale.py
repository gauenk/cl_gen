

# -- path --
import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")
sys.path.append("/home/gauenk/Documents/faiss/contrib/")

# -- python imports --
import cv2
import math,random
import numpy as np
from easydict import EasyDict as edict
from einops import rearrange,repeat
import matplotlib
matplotlib.use("agg")

# -- skimage --
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import chan_vese

# -- pytorch imports --
import torch
import torch.nn.functional as nnF
import torchvision.transforms.functional as tvF
from torchvision.transforms import InterpolationMode as InterpMode

# -- faiss imports --
from nnf_share import padAndTileBatch

# -- project imports --
import settings
from align.interface import get_align_method
from align.xforms import align_from_flow,flow_to_blocks,blocks_to_flow
from pyutils import save_image,images_to_psnrs,KMeans
from datasets import load_dataset
from layers.flownet2_pytorch.utils.flow_utils import flow2img as flow2img_nd

def align_psnr(aligned,isize):
    isize = [isize[k] for k in isize.keys()]
    nframes = len(aligned)
    ref = nframes//2
    psnrs = 0
    aligned = tvF.center_crop(aligned,isize)
    for t in range(nframes):
        if t == ref: continue
        psnrs += np.mean(images_to_psnrs(aligned[t],aligned[ref])).item()
    psnrs /= (nframes-1)
    return psnrs

def flow2img(flow,H,T):
    flow = rearrange(flow,'i (h w) t two -> t i two h w',h=H)
    flow_fmt = flow.type(torch.float).cpu().numpy()
    flow_fmt = rearrange(flow_fmt,'t i two h w -> t i h w two',h=H)
    flow_fmt = flow_fmt[:,0]
    flow_img = [flow2img_nd(flow_fmt[t]) for t in range(T)]
    flow_img = np.stack(flow_img)
    flow_img = torch.FloatTensor(flow_img)
    flow_img = rearrange(flow_img,'t h w c -> t c h w')
    return flow_img

def tiled_to_img(tiled,ps):
    etiled = rearrange(tiled,'t i (ps1 ps2 c) h w -> t i ps1 ps2 c h w',ps1=ps,ps2=ps)
    mid_ps = ps//2
    img = etiled[:,:,mid_ps,mid_ps,:,:,:]
    return img

def cluster_flow(flow,H,nclusters=3):


    # -- compute clusters --
    nimages,hw,nframes,two = flow.shape
    flow = rearrange(flow,'i hw t two -> t (i hw) two')
    flow = flow.type(torch.float).to(0).contiguous()
    clusters,centroids = [],[]
    for t in range(nframes):
        clusters_t, centroids_t, counts, dists = KMeans(flow[[t]],K=nclusters)
        clusters.append(clusters_t)
        centroids.append(centroids_t)
    clusters = torch.stack(clusters)
    print("clusters.shape: ",clusters.shape)
    clusters = rearrange(clusters,'t 1 (h w) -> t 1 h w',h=H)
    # print(clusters.shape)
    # save_image("clusters.png",clusters.type(torch.float))
    # print(clusters)
    
    # -- replace cluster region with most frequent "block" value --
    return clusters

def replace_flow_median(flow,clusters,H,nblocks):
    # -- unpack shape --
    nimages,hw,nframes,two = flow.shape
    blocks = flow_to_blocks(flow,nblocks,ftype="ref")
    blocks = rearrange(blocks,'i (h w) t -> t i h w',h=H)

    # -- create blocks --
    blocks = blocks.cpu().numpy()
    clusters = clusters.cpu().numpy()
    mblocks = np.zeros_like(blocks)
    uniques = list(np.unique(clusters))
    print(uniques)
    for u in uniques:
        indices = np.where(clusters == u)
        iblocks = blocks[indices]
        median = np.median(iblocks)
        mblocks[indices] = median
    mblocks = torch.LongTensor(mblocks)
                            
    # -- create flow --
    mblocks = rearrange(mblocks,'t i h w -> i (h w) t')
    mflow = blocks_to_flow(mblocks,nblocks)

    return mflow

def run_multiscale_nnf(cfg,noisy,clean,nlevels=3,verbose=False):
    T,C,H,W = noisy.shape
    nframes = T
    noisy = noisy[:,None]
    clean = clean[:,None]

    isize = edict({'h':H,'w':W})
    isize_l = [H,W]
    pad3 = cfg.nblocks//2+3//2
    psize3 = edict({'h':H-pad3,'w':W-pad3})
    pad = cfg.nblocks//2+cfg.patchsize//2
    psize = edict({'h':H-pad,'w':W-pad})
    cfg_patchsize = cfg.patchsize

    factor = 2
    noisy = tvF.resize(noisy[:,0],[H//factor,W//factor],
                       interpolation=InterpMode.BILINEAR)[:,None]
    clean = tvF.resize(clean[:,0],[H//factor,W//factor],
                       interpolation=InterpMode.BILINEAR)[:,None]
    T,_,C,H,W = noisy.shape
    isize = edict({'h':H,'w':W})
    isize_l = [H,W]
    pad3 = cfg.nblocks//2+3//2
    psize3 = edict({'h':H-pad3,'w':W-pad3})
    pad = cfg.nblocks//2+cfg.patchsize//2
    psize = edict({'h':H-pad,'w':W-pad})
    cfg_patchsize = cfg.patchsize


    # -- looks good --
    cfg.patchsize = 3
    align_fxn = get_align_method(cfg,"l2_global")
    _,flow = align_fxn(clean.to(0))
    aclean = align_from_flow(clean,flow,cfg.nblocks,isize=isize)
    save_image("aclean.png",aclean)
    apsnr = align_psnr(aclean,psize3)
    print("[global] clean: ",apsnr)

    # -- looks not good --
    _,flow = align_fxn(noisy.to(0))
    isize = edict({'h':H,'w':W})
    aclean = align_from_flow(clean,flow,cfg.nblocks,isize=isize)
    save_image("aclean_rs1.png",aclean)
    apsnr = align_psnr(aclean,psize3)
    print("noisy: ",apsnr)

    # -- fix it --
    cfg.nblocks = 5
    align_fxn = get_align_method(cfg,"pair_l2_local")
    _,flow = align_fxn(aclean.to(0))
    isize = edict({'h':H,'w':W})
    aclean = align_from_flow(aclean,flow,cfg.nblocks,isize=isize)
    save_image("aclean_rs1.png",aclean)
    apsnr = align_psnr(aclean,psize3)
    print("[fixed] noisy: ",apsnr)

    #
    # -- [Tiled] try it again and to fix it --
    #
    img_ps = 3
    cfg.patchsize = img_ps
    cfg.nblocks = 50
    tnoisy = padAndTileBatch(noisy,cfg.patchsize,cfg.nblocks)
    tclean = padAndTileBatch(clean,cfg.patchsize,cfg.nblocks)
    t2i_clean = tvF.center_crop(tiled_to_img(tclean,img_ps),isize_l)
    print(t2i_clean.shape,clean.shape)
    save_image("atiled_to_img.png",t2i_clean)
    delta = torch.sum(torch.abs(clean - t2i_clean)).item()
    assert delta < 1e-8, "tiled to image must work!"

    cfg.patchsize = 3
    align_fxn = get_align_method(cfg,"pair_l2_local")
    _,flow = align_fxn(tnoisy.to(0))
    print(flow.shape,tclean.shape,clean.shape,np.sqrt(flow.shape[1]))

    nbHalf = cfg.nblocks//2
    pisize = edict({'h':H+2*nbHalf,'w':W+2*nbHalf})
    aclean = align_from_flow(tclean,flow,cfg.nblocks,isize=pisize)
    aclean_img = tvF.center_crop(tiled_to_img(aclean,img_ps),isize_l)
    save_image("aclean_rs1_tiled.png",aclean_img)
    apsnr = align_psnr(aclean_img,psize3)
    print("[tiled] noisy: ",apsnr)

    # i want to use a different block size but I need to correct the image padding..?

    # def shrink_search_space(tclean,flow,nblocks_prev,nblocks_curr):
    #     print("tclean.shape: ",tclean.shape)
    #     print("flow.shape: ",flow.shape)
    #     T,_,C,H,W = tclean.shape
    #     flow = rearrange(flow,'i (h w) t two -> t i two h w',h=H)
    #     tclean = tvF.center_crop(tclean,new_size)
    #      = tvF.center_crop(tclean,new_size)
        
    nblocks_prev = cfg.nblocks
    cfg.nblocks = 5
    # tclean,flow = shrink_search_space(tclean,flow,nblocks_prev,cfg.nblocks)
    align_fxn = get_align_method(cfg,"pair_l2_local")
    at_clean = align_from_flow(tclean,flow,cfg.nblocks,isize=pisize)
    _,flow_at = align_fxn(at_clean.to(0))
    aaclean = align_from_flow(at_clean,flow_at,cfg.nblocks,isize=pisize)
    aaclean_img = tvF.center_crop(tiled_to_img(aaclean,img_ps),isize_l)
    save_image("aclean_rs1_fixed.png",aaclean_img)
    apsnr = align_psnr(aaclean_img,psize3)
    print("[fixed] noisy: ",apsnr)

    exit()

    cfg.patchsize = 1#cfg_patchsize

    align_fxn = get_align_method(cfg,"pair_l2_local")
    # clusters = cluster_flow(flow,H,nclusters=4)
    cflow = flow#replace_flow_median(flow,clusters,H,cfg.nblocks)
    # save_image("clusters.png",clusters.type(torch.float))
    cflow_img = flow2img(cflow,H,T)
    save_image("cflow.png",cflow_img)
    aclean = align_from_flow(clean,cflow,cfg.nblocks,isize=isize)
    save_image("aclean_rs1_cf.png",aclean)

    print(cflow[:,64*64+64])
    apsnr = align_psnr(aclean,psize)
    print("noisy_cf: ",apsnr)
    print(flow.shape)

    # flow = rearrange(flow,'i (h w) t two -> t i two h w',h=H)
    # print_stats(flow)
    flow_img = flow2img(flow,H,T)
    save_image("flow.png",flow_img)
    print(torch.histc(flow.type(torch.float)))
    

    factor = 2
    cfg.nblocks = max(cfg.nblocks//2,3)
    cfg.patchsize = 1
    # cfg.patchsize = max(cfg.patchsize//2,3)
    noisy_rs = tvF.resize(noisy[:,0],[H//factor,W//factor],
                       interpolation=InterpMode.BILINEAR)[:,None]
    _,flow_rs = align_fxn(noisy_rs.to(0))


    clean_rs = tvF.resize(clean[:,0],[H//factor,W//factor],
                          interpolation=InterpMode.BILINEAR)[:,None]
    isize = edict({'h':H//factor,'w':W//factor})
    aclean = align_from_flow(clean_rs,flow_rs,cfg.nblocks,isize=isize)
    save_image("aclean_rs2.png",aclean)
    apsnr = align_psnr(aclean,psize)
    print("rs2",apsnr,cfg.nblocks,cfg.patchsize)

    clusters = cluster_flow(flow_rs,H//factor,nclusters=3)
    save_image("clusters_rs.png",clusters.type(torch.float))
    # cflow_rs = cluster_flow(flow_rs,H//factor,nclusters=5)
    # print(cflow_rs)

    aclean = align_from_flow(clean_rs,cflow_rs,cfg.nblocks,isize=isize)
    save_image("aclean_rs2_cl.png",aclean)
    apsnr = align_psnr(aclean,psize)
    print("rs2_cl",apsnr,cfg.nblocks,cfg.patchsize)
    exit()

    print(flow_rs.shape)
    # flow_rs = rearrange(flow_rs,'i (h w) t two -> t i two h w',h=H//factor)
    print(flow_rs.shape)
    flow_img = flow2img(flow_rs,H//factor,T)
    save_image("flow_rs2.png",flow_img)
    fmin,fmax,fmean = print_stats(flow_rs)
    print(torch.histc(flow_rs.type(torch.float),max=50,min=-50))
    

    factor = 4
    cfg.nblocks = max(cfg.nblocks//2,3)
    # cfg.patchsize = max(cfg.patchsize//2,3)
    noisy_rs = tvF.resize(noisy[:,0],[H//factor,W//factor],
                       interpolation=InterpMode.BILINEAR)[:,None]
    _,flow_rs = align_fxn(noisy_rs.to(0))

    clean_rs = tvF.resize(clean[:,0],[H//factor,W//factor],
                          interpolation=InterpMode.BILINEAR)[:,None]
    isize = edict({'h':H//factor,'w':W//factor})
    aclean = align_from_flow(clean_rs,flow_rs,cfg.nblocks,isize=isize)
    save_image("aclean_rs4.png",aclean)
    apsnr = align_psnr(aclean,psize)
    print(apsnr,cfg.nblocks,cfg.patchsize)

    print(flow_rs.shape)
    # flow_rs = rearrange(flow_rs,'i (h w) t two -> t i two h w',h=H//factor)
    print(flow_rs.shape)
    flow_img = flow2img(flow_rs,H//factor,T)
    save_image("flow_rs4.png",flow_img)
    fmin,fmax,fmean = print_stats(flow_rs)
    print(torch.histc(flow_rs.type(torch.float),max=50,min=-50))

    

def print_stats(tensor):
    fmin,fmax = tensor.min().item(),tensor.max().item()
    fmean = tensor.type(torch.float).mean().item()
    print("[min,max,mean]: (%2.2f,%2.2f,%2.2f)" % (fmin,fmax,fmean))
    return fmin,fmax,fmean

def get_cfg_defaults():
    cfg = edict()

    # -- frame info --
    cfg.nframes = 3
    cfg.frame_size = [512,512]
    # cfg.frame_size = [128,128]

    # -- data config --
    cfg.dataset = edict()
    cfg.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.dataset.name = "burst_with_flow_kitti"
    cfg.dataset.dict_loader = True
    cfg.dataset.num_workers = 1
    cfg.set_worker_seed = True
    cfg.batch_size = 1
    cfg.drop_last = {'tr':True,'val':True,'te':True}
    cfg.noise_params = edict({'pn':{'alpha':10.,'std':0},
                              'g':{'std':25.0},'ntype':'g'})
    cfg.dynamic_info = edict()
    cfg.dynamic_info.mode = 'global'
    cfg.dynamic_info.frame_size = cfg.frame_size
    cfg.dynamic_info.nframes = cfg.nframes
    cfg.dynamic_info.ppf = 2
    cfg.dynamic_info.textured = True
    cfg.random_seed = 0

    # -- combo config --
    cfg.nblocks = 5
    cfg.patchsize = 3

    return cfg

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

def test_multiscale():
    
    # -- exp params --
    cfg = get_cfg_defaults()
    cfg.random_seed = 123
    nbatches = 20
    cfg.noise_params.ntype = 'g'
    cfg.noise_params.g.std = 50.
    cfg.patchsize = 3
    cfg.nblocks = 100
    noise_level = cfg.noise_params.g.std
    print("Image Noise Level: %2.3f" % noise_level)

    # -- set random seed --
    # set_seed(cfg.random_seed)	

    # -- load dataset --
    print("load image dataset.")
    data,loaders = load_dataset(cfg,"dynamic")
    tr_iter = iter(data.tr)

    # -- get sample --
    nsamples = 1
    noise_reduction,seg_delta,nnf_quality = [],[],[]
    for i in range(nsamples):
        sample = next(tr_iter)
        noisy = sample['dyn_noisy']
        clean = sample['dyn_clean']

        output = run_multiscale_nnf(cfg,noisy,clean,
                                    nlevels=3,verbose=False)

