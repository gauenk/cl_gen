

# -- python imports --
from easydict import EasyDict as edict
from einops import rearrange,repeat

# -- pytorch imports --
import torch

# -- project deps --
# from pyutils.vst import anscombe
from patch_search import get_score_function
from datasets.transforms import get_noise_transform

# -- self deps --
import align.nnf as nnf
from align.combo import EvalBlockScores,EvalBootBlockScores
from align.combo.optim import AlignOptimizer
from align.xforms import align_from_pix,flow_to_pix,create_isize,pix_to_flow,align_from_flow,flow_to_blocks
from align.nn_loaders import load_flownet_model

# -- faiss deps --
import sys
sys.path.append("/home/gauenk/Documents/faiss/contrib")
from faiss_interface import align_interface

def return_optional(edict,key,default):
    if edict is None: return default
    if key in edict: return edict[key]
    else: return default

def get_align_noise(align_name):
    if align_name == "same": # use the same noisy samples given
        def align_noise_fxn(noisy,clean):
            return noisy
    else: # use a different noisy sample with a different noise level
        apply_noise = get_noise_transform(align_name,noise_only=True)
        def align_noise_fxn(noisy,clean):
            return apply_noise(clean)
    return align_noise_fxn

def get_align_method(cfg,method_name,align_noise=None,comp_align=True):
    if align_noise is None: align_noise = get_align_noise("same")
    if isinstance(align_noise,bool):
        print("WARNING: [align_noise] is bool. Check calling function.")
    if method_name in ["flownetv2","flownet_v2"]:
        return get_align_flownetv2(cfg,align_noise,comp_align)
    elif method_name == "l2_global":
        return get_align_l2_global(cfg,align_noise)
    elif method_name == "pair_l2_local":
        return get_align_pair_l2_local(cfg,align_noise)
    elif method_name == "gt_of":
        return get_align_gt_of(cfg,align_noise)
    elif method_name == "exh_jointly_l2_local":
        return get_align_exh_jointly_l2_local(cfg,align_noise)
    elif method_name == "bp_jointly_l2_local":
        return get_align_bp_jointly_l2_local(cfg,align_noise)
    elif method_name == "n2n":
        return get_align_n2n(cfg,align_noise)
    elif method_name in ["none","sup"]:
        return get_align_none(cfg,align_noise)
    else:
        raise ValueError(f"Uknown nn architecture [{nn_arch}]")

def get_align_flownetv2(cfg,search_img_select,comp_align=True):
    gpuid = cfg.gpuid
    ref_t = cfg.nframes // 2
    flownet = load_flownet_model(cfg)
    def align_fxn(burst,db=None,gt_info=None):
        if db is None: db = burst
        T,B,C,H,W = burst.shape
        isize = edict({'h':H,'w':W})
        # if cfg.noise_params.ntype == "pn" or cfg.use_anscombe:
        #     search = anscombe.forward(burst)
        #     search -= search.min()
        #     search /= search.max()

        burst = burst.to(gpuid,non_blocking=True)
        clean = return_optional(gt_info,'clean',None)
        search = search_img_select(burst,burst)

        flow = flownet.burst2flow(search).cpu()
        flow = rearrange(flow,'t i h w two -> i (h w) t two')
        aligned = None
        if comp_align:
            aligned = align_from_flow(burst,flow,cfg.pads,isize=isize)
            aligned = aligned.to(burst.device,non_blocking=True)
        return aligned,flow
    return align_fxn

def get_align_l2_global(cfg,search_img_select):
    ref_t = cfg.nframes // 2
    def align_fxn(burst,db=None,gt_info=None):
        if db is None: db = burst
        T,B,C,H,W = burst.shape
        isize = edict({'h':H,'w':W})
        # burst = inputs['burst']
        # isize = inputs['isize']
        # if cfg.noise_params.ntype == "pn" or cfg.use_anscombe:
        #     search = anscombe.forward(burst)
        #     search -= search.min()
        #     search /= search.max()
        clean = return_optional(gt_info,'clean',None)
        search = search_img_select(burst,clean)

        nnf_vals,nnf_pix = nnf.compute_burst_nnf(search,ref_t,cfg.patchsize,K=1)
        shape_str = 't b h w two -> b (h w) t two'
        nnf_pix_best = torch.LongTensor(rearrange(nnf_pix[...,0,:],shape_str))
        flow = pix_to_flow(nnf_pix_best)
        aligned = align_from_flow(burst,flow,0,isize=isize)
        aligned = aligned.to(burst.device,non_blocking=True)
        return aligned,flow
    return align_fxn

def get_align_pair_l2_local(cfg,align_noise):
    runPairSearch = align_interface("pair_l2_local")
    valMode = return_optional(cfg,"offset",0.)
    def align_fxn(burst,db=None,gt_info=None):
        if db is None: db = burst
        T,B,C,H,W = burst.shape
        isize = edict({'h':H,'w':W})

        clean = return_optional(gt_info,'clean',None)
        search = align_noise(burst,clean)
        _,pix = runPairSearch(search,cfg.patchsize,cfg.nblocks,
                              k=1,valMean=valMode)
        pix = rearrange(pix,'t i h w 1 two -> i (h w) t two')
        aligned = align_from_pix(burst,pix,cfg.nblocks)
        aligned = aligned.to(burst.device,non_blocking=True)
        flow = pix_to_flow(pix)
        return aligned,flow

    return align_fxn

def get_align_exh_jointly_l2_local(cfg,align_noise):
    runJointSearch = align_interface("exh_jointly_l2_local")
    valMode = return_optional(cfg,"offset",0.)
    def align_fxn(burst,db=None,gt_info=None):
        if db is None: db = burst
        T,B,C,H,W = burst.shape
        isize = edict({'h':H,'w':W})

        clean = return_optional(gt_info,'clean',None)
        search = align_noise(burst,clean)
        # if cfg.noise_params.ntype == "pn" or cfg.use_anscombe:
        #     search = anscombe.forward(burst)
        #     search -= search.min()
        #     search /= search.max()
        _,flow,_ = runJointSearch(search,cfg.patchsize,cfg.nblocks,
                                  k=1,valMean=valMode)
        flow = rearrange(flow,'t i h w 1 two -> i (h w) t two')
        aligned = align_from_flow(burst,flow,cfg.nblocks,isize=isize)
        aligned = aligned.to(burst.device,non_blocking=True)
        return aligned,flow
    return align_fxn

def get_align_bp_jointly_l2_local(cfg,align_noise):
    runJointSearch = align_interface("bp_jointly_l2_local")
    valMode = return_optional(cfg,"offset",0.)
    def align_fxn(burst,db=None,gt_info=None):
        if db is None: db = burst
        T,B,C,H,W = burst.shape
        isize = edict({'h':H,'w':W})
        # if cfg.noise_params.ntype == "pn" or cfg.use_anscombe:
        #     search = anscombe.forward(burst)
        #     search -= search.min()
        #     search /= search.max()
        clean = return_optional(gt_info,'clean',None)
        search = align_noise(burst,clean)

        _,flow,_ = runJointSearch(search,cfg.patchsize,cfg.nblocks,
                                  k=1,valMean=valMode)
        flow = rearrange(flow,'t i h w 1 two -> i (h w) t two')
        aligned = align_from_flow(burst,flow,cfg.nblocks,isize=isize)
        aligned = aligned.to(burst.device,non_blocking=True)
        return aligned,flow
    return align_fxn

def get_align_gt_of(cfg,align_noise):
    def align_fxn(inputs,db=None,gt_info=None):
        if db is None: db = inputs
        burst = inputs
        flow = gt_info['flow']
        isize = gt_info['isize']
        nimages,npix,nframes_m1,two = flow.shape
        aligned = align_from_flow(burst,flow,0,isize=isize)
        aligned = aligned.to(burst.device,non_blocking=True)
        return aligned,flow
    return align_fxn

def get_align_n2n(cfg,align_noise):
    def align_fxn(inputs,db=None,gt_info=None):
        if db is None: db = inputs
        burst = inputs
        t = burst.shape[0]
        clean = gt_info['clean']
        flow = gt_info['flow']
        noisy2 = align_noise(None,clean)
        noisy2 = torch.stack([burst[t//2],noisy2[t//2]],0)
        return noisy2,flow
    return align_fxn

def get_align_none(cfg,align_noise):
    def align_fxn(inputs,db=None,gt_info=None):
        if db is None: db = inputs
        burst = inputs
        flow = gt_info['flow']
        return burst,flow
    return align_fxn

