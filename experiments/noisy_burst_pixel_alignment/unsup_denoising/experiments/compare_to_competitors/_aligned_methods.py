import torch
from easydict import EasyDict as edict
from einops import rearrange,repeat


from pyutils.vst import anscombe

from patch_search import get_score_function
import align.nnf as nnf
from align.combo.eval_scores import EvalBlockScores
from align.combo.optim import AlignOptimizer
from align.xforms import align_from_pix,flow_to_pix,create_isize,pix_to_flow,align_from_flow,flow_to_blocks

def get_align_method(cfg,method_name):
    if method_name == "l2_global":
        return get_sim_l2_global(cfg)
    elif method_name == "l2_local":
        return get_sim_l2_local(cfg)
    elif method_name == "bs_local_v3":
        return get_sim_bs_local(cfg,"v3")
    elif method_name == "bs_local_v2":
        return get_sim_bs_local(cfg,"v2")
    elif method_name == "of":
        return get_sim_of(cfg)
    else:
        raise ValueError(f"Uknown nn architecture [{nn_arch}]")

def get_sim_l2_global(cfg):
    ref_t = cfg.nframes // 2
    def align_fxn(burst,db=None,gt_info=None):
        if db is None: db = burst
        T,B,C,H,W = burst.shape
        isize = edict({'h':H,'w':W})
        # burst = inputs['burst']
        # isize = inputs['isize']
        search = burst
        if cfg.noise_params.ntype == "pn" or cfg.use_anscombe:
            search = anscombe.forward(burst)
            search /= search.mean()
        nnf_vals,nnf_pix = nnf.compute_burst_nnf(search,ref_t,cfg.patchsize,K=1)
        shape_str = 't b h w two -> b (h w) t two'
        nnf_pix_best = torch.LongTensor(rearrange(nnf_pix[...,0,:],shape_str))
        flow = pix_to_flow(nnf_pix_best)
        aligned = align_from_flow(burst,flow,cfg.nblocks,isize=isize)
        aligned = aligned.to(burst.device,non_blocking=True)
        return aligned,flow
    return align_fxn
    
def get_sim_l2_local(cfg):
    ave_fxn = get_score_function("ave")
    block_batchsize = 128
    eval_block = EvalBlockScores(ave_fxn,"ave",cfg.patchsize,block_batchsize,None)    
    optim = AlignOptimizer("v3")
    iterations,subsizes,K = 0,[],1
    def align_fxn(burst,db=None,gt_info=None):
        if db is None: db = burst
        T,B,C,H,W = burst.shape
        isize = edict({'h':H,'w':W})
        search = burst
        if cfg.noise_params.ntype == "pn" or cfg.use_anscombe:
            search = anscombe.forward(burst)
            search /= search.mean()
        flow = optim.run(search,cfg.patchsize,eval_block,
                        cfg.nblocks,iterations,subsizes,K)
        aligned = align_from_flow(burst,flow,cfg.nblocks,isize=isize)
        aligned = aligned.to(burst.device,non_blocking=True)
        return aligned,flow

    return align_fxn

def get_sim_bs_local(cfg,version): 
    bs_fxn = get_bootstrapping_fxn(version)
    K,subsizes = get_bootstrapping_params(cfg.nframes,cfg.nblocks)
    block_batchsize = 64
    eval_prop = EvalBlockScores(bs_fxn,"bs",cfg.patchsize,block_batchsize,None)    
    optim = AlignOptimizer("v3")
    iterations = 1
    def align_fxn(burst,db=None,gt_info=None):
        if db is None: db = burst
        T,B,C,H,W = burst.shape
        isize = edict({'h':H,'w':W})
        search = burst
        if cfg.noise_params.ntype == "pn" or cfg.use_anscombe:
            search = anscombe.forward(burst)
            search /= search.mean()
        flow = optim.run(search,cfg.patchsize,eval_prop,
                         cfg.nblocks,iterations,subsizes,K)
        aligned = align_from_flow(burst,flow,cfg.nblocks,isize=isize)
        aligned = aligned.to(burst.device,non_blocking=True)
        return aligned,flow
    return align_fxn

def get_bootstrapping_params(nframes,nblocks):
    if nframes == 3:
        K = nblocks**2
        subsizes = [nframes]
    elif nframes == 5:
        K = 2*nblocks
        subsizes = [nframes]
    elif nframes <= 20:
        K = 2*nblocks
        subsizes = [2,]*nframes
    else:
        K = nblocks
        subsizes = [2,]*nframes
    return K,subsizes

def get_bootstrapping_fxn(version):
    if version == "v3":
        score_fxn = get_score_function("bootstrapping_mod3")
    elif version == "v2":
        score_fxn = get_score_function("bootstrapping_mod2")
    else:
        raise ValueError(f"Uknown bootstrapping version [{version}]")
    return score_fxn

def get_sim_of(cfg):
    def align_fxn(inputs,db=None,gt_info=None):
        if db is None: db = burst
        burst = inputs['burst']
        flow = inputs['flow']
        isize = inputs['isize']
        nimages,nframes_m1,two = flow.shape
        aligned = align_from_flow(burst,flow_gt,cfg.nblocks,isize=isize)
        aligned = aligned.to(burst.device,non_blocking=True)
        return aligned,flow
    return align_fxn


