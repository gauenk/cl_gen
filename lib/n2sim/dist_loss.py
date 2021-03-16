
# -- python imports --
import bm3d,uuid,cv2
import numpy as np
import pandas as pd
import numpy.random as npr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from einops import rearrange, repeat, reduce

# -- just to draw an fing arrow --
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


# -- pytorch imports --
import torch
import torch.nn.functional as F
import torchvision.utils as tv_utils

# -- project code --
import settings
from pyutils.timer import Timer
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr
from datasets.transforms import ScaleZeroMean
from layers.ot_pytorch import sink_stabilized

def ot_gaussian_bp(residuals,std=25,reg=1.0,K=4):
    """
    :param residuals: shape [B D C]
    """
    
    # -- init --
    B,D,C = residuals.shape

    # -- grab some batches --
    Bgrid = torch.randperm(B)[:K]

    # -- compute losses --
    ot_loss = 0
    for b in Bgrid:
        rb = residuals[b]
        noise = torch.normal(torch.zeros_like(rb),std=std/255.)
        M = torch.sum(torch.pow(rb.unsqueeze(1) - noise,2),dim=-1)
        loss = sink_stabilized(M,reg)
        ot_loss += loss
    return ot_loss / len(Bgrid)
    
def ot_pairwise2gaussian_bp(residuals,std=25,reg=1.0,K=3):
    """
    :param residuals: shape [B N D C]
    """
    
    # -- init --
    B,N,D,C = residuals.shape

    # -- compute all ot --
    results = ot_all_gaussian_items(residuals,std=std,reg=reg)
    
    # -- create triplets -- 
    indices = get_ot_topK(results,K)

    # -- compute losses --
    ot_loss = 0
    for (b,n) in indices:
        r = residuals[b,n]
        noise = torch.normal(torch.zeros_like(r),std=std/255)
        M = torch.sum(torch.pow(r.unsqueeze(1) - noise,2),dim=-1)
        loss = sink_stabilized(M,reg,print_me=False,print_period=5)
        ot_loss += loss
    return ot_loss / len(indices)

def ot_pairwise_bp(residuals,reg=1.0,K=3):
    """
    :param residuals: shape [B N D C]
    """
    
    # -- init --
    B,N,D,C = residuals.shape

    # -- compute all ot --
    results = ot_all_pairwise_items(residuals,reg=1.0)

    # -- create triplets -- 
    indices = get_ot_topK(results,K)

    # -- compute losses --
    ot_loss = 0
    for (bi,bj,i,j) in indices:
        ri,rj = residuals[bi,i],residuals[bj,j]
        M = torch.sum(torch.pow(ri.unsqueeze(1) - rj,2),dim=-1)
        loss = sink_stabilized(M,reg)
        ot_loss += loss
    return ot_loss / len(indices)

def get_ot_topK(results,K):
    results = results.sort_values('loss',ascending=False)
    # print(np.sum(results['loss'][:K])/results['loss'].sum())
    return results['indices'][:K]

def ot_all_gaussian_items(residuals_raw,std=25,reg=1.0):
    """
    :param residuals: shape [B N D C]
    """
    
    # -- init --
    residuals = residuals_raw.detach().requires_grad_(False)
    B,N,D,C = residuals.shape

    # -- compute losses --
    results = pd.DataFrame({'indices':[],'loss':[]})
    for b in range(B):
        for n in range(N):
            r = residuals[b,n]
            noise = torch.normal(torch.zeros_like(r),std=std/255.)
            M = torch.sum(torch.pow(r.unsqueeze(1) - noise,2),dim=-1)
            loss = sink_stabilized(M,reg)
            result = {'indices':(b,n),'loss':loss}
            results = results.append(result,ignore_index=True)
    return results
    
def ot_all_pairwise_items(residuals_raw,reg=1.0,S=6):
    """
    :param residuals: shape [B N D C]
    """
    
    # -- init --
    residuals = residuals_raw.detach().requires_grad_(False)
    B,N,D,C = residuals.shape

    # -- restrict number of terms  --
    Ngrid_i,Ngrid_j = np.tril_indices(N)
    order = npr.permutation(len(Ngrid_i))
    Ngrid_i,Ngrid_j = Ngrid_i[order[:S]],Ngrid_j[order[:S]]

    # -- compute losses --
    results = pd.DataFrame({'indices':[],'loss':[]})
    for bi in range(B):
        for bj in range(B):
            if bi > bj: continue
            for ni,nj in zip(Ngrid_i,Ngrid_j):
                ri,rj = residuals[bi,ni],residuals[bj,nj]
                M = torch.sum(torch.pow(ri.unsqueeze(1) - rj,2),dim=-1)                    
                loss = sink_stabilized(M,reg)
                result = {'indices':(bi,bj,ni,nj),'loss':loss}
                results = results.append(result,ignore_index=True)
    return results


def compute_ot_frame(aligned,rec,raw,reg=0.5):
    # -- init --
    B,N,C,H,W = aligned.shape
    aligned = aligned.detach()
    rec = rec.detach()

    # -- rec residuals --
    rec_res = aligned - rec.unsqueeze(1).repeat(1,N,1,1,1)
    rec_res = rearrange(rec_res,'b n c h w -> b n (h w) c')
    ot_loss_rec,ot_loss_rec_mid,ot_loss_rec_w = ot_frame_pairwise(rec_res,reg=reg)

    # -- raw residuals --
    raw_res = aligned - raw.unsqueeze(1).repeat(1,N,1,1,1)
    raw_res = rearrange(raw_res,'b n c h w -> b n (h w) c')
    ot_loss_raw,ot_loss_raw_mid,ot_loss_raw_w = ot_frame_pairwise(raw_res,reg=reg)

    # -- format results --
    results = {'ot_loss_rec_frame':ot_loss_rec,
               'ot_loss_rec_frame_mid':ot_loss_rec_mid,
               'ot_loss_rec_frame_w':ot_loss_rec_w,
               'ot_loss_raw_frame':ot_loss_raw,
               'ot_loss_raw_frame_mid':ot_loss_raw_mid,
               'ot_loss_raw_frame_w':ot_loss_raw_w,}

    return results

def ot_frame_pairwise(residuals,reg=0.5):
    """
    :paraam residuals: shape [B N D C]
    """
    
    # -- init --
    B,N,D,C = residuals.shape

    # -- create pairs --
    pairs,S = select_pairs(N)

    # -- compute losses --
    ot_loss,ot_loss_mid,ot_loss_w = 0,0,0
    for b in range(B):
        for (i,j) in pairs:
            ri,rj = residuals[b,i],residuals[b,j]
            M = torch.sum(torch.pow(ri.unsqueeze(1) - rj,2),dim=-1)
            loss = sink_stabilized(M,reg)
            ot_loss += loss
            weight = (torch.sum(ri**2) + torch.sum(rj**2)).item()
            ot_loss_w += weight * loss
            if i == N//2 or j == N//2:
                ot_loss_mid += loss
    return ot_loss,ot_loss_mid,ot_loss_w
    

def compute_ot_burst(aligned,rec,raw,reg=0.5):
    # -- init --
    B,N,C,H,W = aligned.shape
    aligned = aligned.detach()
    rec = rec.detach()

    # -- rec residuals --
    rec_res = aligned - rec.unsqueeze(1).repeat(1,N,1,1,1)
    rec_res = rearrange(rec_res,'b n c h w -> b (n h w) c')
    ot_loss_rec,ot_loss_rec_w = ot_burst_pairwise(rec_res,reg=reg)

    # -- raw residuals --
    raw_res = aligned - raw.unsqueeze(1).repeat(1,N,1,1,1)
    raw_res = rearrange(raw_res,'b n c h w -> b (n h w) c')
    ot_loss_raw,ot_loss_raw_w = ot_burst_pairwise(raw_res,reg=reg)

    # -- format results --
    results = {'ot_loss_rec_burst':ot_loss_rec,'ot_loss_raw_burst':ot_loss_raw,
               'ot_loss_rec_burst_w':ot_loss_rec_w,'ot_loss_raw_burst_w':ot_loss_raw_w,}

    return results

def ot_burst_pairwise(residuals,reg=0.5):
    """
    :param residuals: shape [B D C]
    """
    # -- init --
    B,D,C = residuals.shape

    # -- create pairs --
    pairs,S = select_pairs(B)

    # -- compute losses --
    ot_loss,ot_loss_w = 0,0
    for (i,j) in pairs:
        ri,rj = residuals[i],residuals[j]
        M = torch.sum(torch.pow(ri.unsqueeze(1) - rj,2),dim=-1)
        loss = sink_stabilized(M,reg)
        ot_loss += loss
        weight = (torch.sum(ri**2) + torch.sum(rj**2)).item()
        ot_loss_w += weight * loss
    return ot_loss,ot_loss_w

def select_pairs( N, S = None, skip_middle = False):
    if skip_middle:
        pairs = list(set([(i,j) for i in range(N) for j in range(N) if i != N//2 and j != N//2]))
    else:
        pairs = list(set([(i,j) for i in range(N) for j in range(N)]))
    P = len(pairs)
    if S is None: S = P
    r_idx = npr.choice(range(P),S)
    s_pairs = [pairs[idx] for idx in r_idx]
    return s_pairs,S

def create_middle_loop_indices(B,N,S):
    indices = []
    for i in range(N):
        for bi in range(B):
            for bj in range(B):
                if bi > bj: continue
                index = (bi,bj,i,N//2)
                indices.append(index)

    P = len(indices)
    indices = list(set(indices))
    assert P == len(indices), "We only created the list with unique elements"
    if S is None: S = P
    if S > P: r_idx = npr.choice(range(P),P)
    else: r_idx = npr.choice(range(P),S)
    s_indices = [indices[idx] for idx in r_idx]
    return s_indices,S

def create_no_middle_loop_indices(B,N,S):
    indices = []
    for i in range(N):
        for j in range(N):
            if i > j: continue
            if i == N//2 or j == N//2: continue
            for bi in range(B):
                for bj in range(B):
                    if bi > bj: continue
                    index = (bi,bj,i,j)
                    indices.append(index)

    P = len(indices)
    indices = list(set(indices))
    assert P == len(indices), "We only created the list with unique elements"
    if S is None: S = P
    if S > P: r_idx = npr.choice(range(P),P)
    else: r_idx = npr.choice(range(P),S)
    s_indices = [indices[idx] for idx in r_idx]
    return s_indices,S
    
def create_loop_indices(B,N,S):
    indices = []
    for i in range(N):
        for j in range(N):
            if i > j: continue
            for bi in range(B):
                for bj in range(B):
                    if bi > bj: continue
                    index = (bi,bj,i,j)
                    indices.append(index)

    P = len(indices)
    indices = list(set(indices))
    assert P == len(indices), "We only created the list with unique elements"
    if S is None: S = P
    if S > P: r_idx = npr.choice(range(P),P)
    else: r_idx = npr.choice(range(P),S)
    s_indices = [indices[idx] for idx in r_idx]
    return s_indices,S

def create_triplets(pairs,B):
    triplets = []
    for b in range(B):
        for (i,_) in pairs:
            for (_,j) in pairs:
                triplets.append([b,i,j])
    return triplets


def create_arrow_image(directions,pad=2):
    D = len(directions)
    assert D == 1,"Only one direction right now."
    W = 100
    S = (W + pad) * D + pad
    arrows = np.zeros((S,W+2*pad,3))
    direction = directions[0]
    for i in range(D):
        col_i = (pad+W)*i+pad
        canvas = arrows[col_i:col_i+W,pad:pad+W,:]
        start_point = (0,0)
        x_end = direction[0].item()
        y_end = direction[1].item()
        end_point = (x_end,y_end)

        fig = Figure(dpi=300)
        plt_canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.annotate("",
                    xy=end_point, xycoords='data',
                    xytext=start_point, textcoords='data',
                    arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),
        )
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        plt_canvas.draw()       # draw the canvas, cache the renderer
        canvas = np.array(plt_canvas.buffer_rgba())[:,:,:]
        arrows = canvas
    arrows = torch.Tensor(arrows.astype(np.uint8)).transpose(0,2).transpose(1,2)
    return arrows

def sparse_filter_loss(filters):
    """
    :params filters: shape: [B,N,K2,1,H,W] with H = 1 or H
    """

    # -- init --
    B,N,K2 = filters.shape[:3]
    filters = filters.view(B,N,K2)
    afilters = torch.abs(filters)

    # -- standard l1 loss --
    l1_filters = torch.sum(afilters)

    # -- penalize filter values not close to 1 --
    to1_filters = torch.mean(torch.abs(torch.sum(afilters,dim=2) - 1))

    # -- old sparse filter loss --
    # afilters = torch.abs(filters)
    # sfilters = torch.mean(afilters,dim=2)
    # zero = torch.FloatTensor([0.]).to(filters.device)
    # wfilters = torch.where(sfilters > float(1.), sfilters, zero)
    
    loss = l1_filters + to1_filters
    return loss


def w_gaussian_bp(residuals,raw_std=25):

    # -- target info --
    tgt_mean = 0.
    tgt_std = raw_std / 255.

    # -- residual info --
    res_mean = residuals.mean()
    res_std = residuals.std()

    # -- final loss --
    w_loss = ( tgt_mean - res_mean )**2 + ( tgt_std - res_std )**2

    return torch.exp(w_loss)

def kl_gaussian_bp(residuals,raw_std=25):

    # -- target info --
    tgt_mean = 0.
    tgt_std = raw_std / 255.

    # -- residual info --
    res_mean = residuals.mean()
    res_std = residuals.std()

    # -- each term --
    eps = 1e-15
    kl_loss_term1 = torch.log( res_std / tgt_std  + eps)
    kl_loss_term2 = ( tgt_std**2 + (tgt_mean - res_mean)**2 ) / ( 2 * res_std**2 )
    kl_loss_term3 = 0.5

    # -- final loss --
    kl_loss = kl_loss_term1 + kl_loss_term2 - kl_loss_term3

    return torch.exp(kl_loss)


def kl_pairwise_bp(residuals,bins=255,K=10,supervised=True):
    """
    :param residuals: shape [B N D C]
    """
    
    # -- init --
    B,N,D,C = residuals.shape

    # -- create triplets
    S = B*K
    indices,S = create_loop_indices(B,N,S)
    # indices,S = create_no_middle_loop_indices(B,N,S)
    # if supervised:
    #     indices,S = create_middle_loop_indices(B,N,S)
    # else:
    #     indices,S = create_no_middle_loop_indices(B,N,S)
        # indices,S = create_loop_indices(B,N,S)
    
    # -- compute losses --
    kl_loss = 0
    for (bi,bj,i,j) in indices:
        ri,rj = residuals[bi,i],residuals[bj,j]
        kl_loss += compute_binned_kl(ri,rj,bins=255)
    return kl_loss / len(indices)

def compute_binned_kl(ri,rj,bins=255):
    """
    compute empirical kl for binned residuals
    """
    ri,rj = ri.reshape(-1),rj.reshape(-1)

    amin = min([ri.min(),rj.min()])
    ri -= amin
    rj -= amin
    amax = max([ri.max(),rj.max()])
    scale = bins/amax

    ri = torch.floor(scale*ri).int()
    rj = torch.floor(scale*rj).int()

    cri = torch.bincount(ri).float()
    crj = torch.bincount(rj).float()

    cri /= cri.sum()
    crj /= crj.sum()

    plot_historgram_pair(cri,crj,rand_name=True)
    
    si = int(cri.size()[0])
    sj = int(crj.size()[0])

    if si > sj: cri = cri[:sj]
    elif si < sj: crj = crj[:si]
    args = torch.where(torch.logical_and(cri,crj))[0]

    
    kl = 0
    for index in args:
        freq_index_i = cri[index]
        freq_index_j = crj[index]
        kl += freq_index_i * (torch.log(freq_index_i) - torch.log(freq_index_j))
    return kl


def create_loop_indices(B,N,S):
    indices = []
    for i in range(N):
        for j in range(N):
            if i > j: continue
            for bi in range(B):
                for bj in range(B):
                    if bi > bj: continue
                    index = (bi,bj,i,j)
                    indices.append(index)

    P = len(indices)
    indices = list(set(indices))
    assert P == len(indices), "We only created the list with unique elements"
    if S is None: S = P
    if S > P: r_idx = npr.choice(range(P),P)
    else: r_idx = npr.choice(range(P),S)
    s_indices = [indices[idx] for idx in r_idx]
    return s_indices,S


