
# -- python imports --
import math
import numpy as np
import heapq
import ot.gpu as otg

# -- numba imports --
from numba import jit, cuda, float32

# -- pytorch imports --
import torch
import torch.multiprocessing as mp

# -- [local] project imports --
from .pair_ot import compute_ot_mats

def search_raw_array_numba(res,noise_level,db,query,K):
    db_numpy = db.numpy()
    query_numpy = query.numpy()

    D,I = emd_search_raw_array_numba(db_numpy,query_numpy,K,300,300.)
    return D.cpu(),I.cpu()

def compute_matrix_hists(mat,nbins,maxp):
    eps = 1e-16
    nmlz_mat = mat*maxp
    S,D = mat.shape
    matHists = np.zeros( ( S, nbins ) )
    for s in range(S):
        matHists[s] = np.histogram(nmlz_mat[s],bins=nbins,range=[0.,maxp])[0].astype(np.float)
        matHists[s] /= (np.sum(matHists[s]) + eps)
    matHists = torch.FloatTensor(matHists)
    return matHists

def emd_search_raw_array_numba(db,query,K,hist_nbins,hist_maxp):
    # -- shapes --
    D,P = db.shape
    Q,P = query.shape

    # -- init outputs --
    outDist = torch.zeros( (Q, K), dtype=torch.float32, device='cuda:0')
    outIndex = torch.zeros( (Q, K), dtype=torch.long, device='cuda:0')

    # -- compute histograms of database --
    # dbHists = np.zeros( (D,hist_nbins), dtype=np.float32 )
    dbHists = compute_matrix_hists(db,hist_nbins,hist_maxp)
    # compute_matrix_hists(db,dbHists)
    qHists = compute_matrix_hists(query,hist_nbins,hist_maxp)
    # qHists = np.zeros( (Q,hist_nbins), dtype=np.float32 )
    # compute_matrix_hists(db,qHists)
    grid = np.arange(hist_nbins,dtype=float)[:,None]
    M = torch.FloatTensor(otg.dist(grid,grid))
    M /= torch.max(M)

    # -- to cuda --
    dbHists = dbHists.cuda(non_blocking=True)
    qHists = qHists.cuda(non_blocking=True)
    M = M.cuda(non_blocking=True)

    # -- tilings --
    numTileQ = 8 
    tileQ = Q // numTileQ

    use_mp = False
    if use_mp:
        # -- share memory --
        outDist = outDist.share_memory_()
        outIndex = outIndex.share_memory_()
        dbHists = dbHists.share_memory_()
        qHists = qHists.share_memory_()
        M = M.share_memory_()
        args = (qHists,dbHists,outDist,outIndex,M,K,tileQ)
        mp.spawn(mp_tileQ_search,args=args,nprocs=numTileQ,join=True)
    else:
        for tile_q in range(0,Q,tileQ):
            q_size = min([tileQ,Q-tile_q])
            q_start = tile_q
            q_end = q_start + q_size
    
            queryView = qHists[q_start:q_end]
            outDistBufRowView = outDist[q_start:q_end]
            outIndexBufRowView = outIndex[q_start:q_end]
    
            compute_ot_mats(q_start,queryView,dbHists,M,outDistBufRowView,outIndexBufRowView,K)
    return outDist,outIndex

def mp_tileQ_search(proc_index,qHists,dbHists,outDist,outIndex,M,K,tileQ):
    Q,H = qHists.shape

    # -- indexing setup --
    tile_q = proc_index * tileQ
    q_size = min([tileQ,Q-tile_q])
    q_start = tile_q
    q_end = q_start + q_size

    # -- subset to view --
    queryView = qHists[q_start:q_end]
    outDistBufRowView = outDist[q_start:q_end]
    outIndexBufRowView = outIndex[q_start:q_end]

    # -- run ot alg
    compute_ot_mats(queryView,dbHists,M,outDistBufRowView,outIndexBufRowView,K)
    
