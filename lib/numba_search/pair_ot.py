
# -- python imports --
import numpy as np

# -- pytorch imports --
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.transforms.functional as tvF

# -- numba --
from numba import jit, cuda, float32


# -- ot computation --
import ot
import ot.gpu as otg

eps = 1e-16
reg = 1e-3

nbins = 300
maxp = 300.
grid = np.arange(nbins,dtype=float)[:,None]
W = otg.dist(grid,grid)


def compute_ot(data_a,data_b):

    nmlz_a = data_a.numpy()*255.
    nmlz_b = data_b.numpy()*255.

    hist_a = np.histogram(nmlz_a,bins=nbins,range=[0.,maxp])[0].astype(np.float)
    hist_b = np.histogram(nmlz_b,bins=nbins,range=[0.,maxp])[0].astype(np.float)

    # -- prepare for sinkhorn --
    hist_a /= np.sum(hist_a)
    hist_b /= np.sum(hist_b)

    # -- remove locations by non-zero --
    nz = np.where( np.logical_or(hist_a > 0, hist_b > 0) )[0]
    hist_a_sub = hist_a[nz] + eps
    hist_b_sub = hist_b[nz] + eps

    # -- take subset of histogram distances --
    xnz,ynz = np.meshgrid(nz,nz)
    W_sub = W[xnz,ynz]
    W_sub /= W_sub.max()

    # -- run sinkhorn --
    Mat = otg.sinkhorn(hist_a_sub,hist_b_sub,W_sub,reg)
    d = np.sum(W_sub * Mat)
    return d

def compute_ot_mats(qStartIndex,query_hists,db_hists,M,outDists,outIndex,K):

    # -- consts --
    reg = 1e2
    eps = 1e-16 #1e-16
    tol = 1e-15
    Q,H = query_hists.shape
    D,H = db_hists.shape

    # -- exploit local information for faster computation --
    R = 5
    W = 256
    qindex_grid = torch.arange(0,256*256).reshape(256,256)

    for q in range(Q):
        query_hist = query_hists[q]

        # -- window select --
        # qwindow = torch.arange(q-100,q+100)
        qRow,qCol = (q+qStartIndex) // W, q % W
        top,left = max([qRow - R, 0]),max([qCol - R,0])
        qwindow = tvF.crop(qindex_grid,top,left,2*R,2*R).reshape(-1)
        # qwindow = torch.arange(0,D)
        db_hists_view = db_hists[qwindow]
        # db_hists_view = torch.cat([query_hist[None,:],db_hists_view],dim=0)
        # print(db_hists_view.shape)
        # -- compute distances --
        distances = l2_hists(M,query_hist,db_hists_view,eps,reg,tol)
        # distances = lightspeed_sink(M,query_hist,db_hists_view,eps,reg,tol)

        # -- top k --
        indices = torch.argsort(distances)[:K].long()
        outIndex[q,:] = qwindow[indices]
        outDists[q,:] = distances[indices]

        #
        # -- testing --
        #

        # mid = len(qwindow)//2+R
        # print(outDists[q,0].item(),indices[0].item(),outIndex[q,0].item(),distances[mid].item(),mid,q+qStartIndex)

        # distances = run_otg_sink(M,query_hist,db_hists_view,1./eps,reg,tol)
        # -- top k --
        # indices = torch.argsort(distances)[:K].long()
        # outIndex[q,:] = qwindow[indices]
        # outDists[q,:] = distances[indices]
        # print("run_ot_sink",outDists[q,0].item(),indices[0].item(),outIndex[q,0].item(),distances[mid].item(),mid,q+qStartIndex)
        # have: a,b,c, d,e,f
        # want: a == d,       b == e,              c == f
        # means: min dist eq, verifid ID map for local ID, verified ID for larger array

        # abs_diff = np.abs(outIndex[q,0].item() - (q+qStartIndex))
        # if abs_diff != 0: print(q,q+qStartIndex,abs_diff)
        # if q >= 1633:
        #     print(q,outIndex[q,:])
        #     print(qRow,qCol)
        # if q > 1640:
        #     hist = query_hists[1634]
        #     nz = hist.nonzero(as_tuple=True)[0]
        #     print('q_1634',query_hists[1634][nz],nz)
        #     print('db_1634',db_hists[1634])
        #     print('q_1633',db_hists[1633])
        #     print('db_1633',db_hists[1633])
        #     exit()
        # print(outDists[q,:])
        # print("Query: %d/%d" % (q,Q) )

        #
        # run it again
        #
        # qwindow = torch.arange(0,D)
        # db_hists_view = db_hists[qwindow]
        # # -- compute distances --
        # distances = l2_hists(M,query_hist,db_hists_view,eps,reg,tol)
        # # distances = lightspeed_sink(M,query_hist,db_hists_view,eps,reg,tol)
        # # -- top k --
        # indices = torch.argsort(distances)[:K].long()
        # outIndex[q,:] = qwindow[indices]
        # print("b",outIndex[q,:])
        # outDists[q,:] = distances[indices]
        # print(outDists[q,:])
        # print("Query: %d/%d" % (q,Q) )

def l2_hists(M,query_hist,db_hists,eps,reg,tol,verbose=False):
    D,H = db_hists.shape
    rq = query_hist[None,:].repeat(D,1)
    distances = torch.sum(F.mse_loss(rq,db_hists,reduction='none'),dim=1)
    return distances

def run_otg_sink(Mfull,query_hist,db_hists,eps,reg,tol):
    # -- remove locations by non-zero --
    nz = torch.where( query_hist > 0 )[0]
    nz = query_hist.nonzero(as_tuple=True)[0]
    query_hist = query_hist[nz]
    M = Mfull[nz,:]
    N,D = db_hists.shape
    d = torch.FloatTensor(otg.bregman.sinkhorn(query_hist,db_hists.T,M,reg,
                                               method='sinkhorn_stabilized'))
    # d = []
    # for n in range(N):
    #     result = otg.bregman.sinkhorn_knopp(query_hist,db_hists[n],M,reg)
    #     print(result)
    #     d.append(result)
    # d = torch.cat(d,dim=0)
    return d

def lightspeed_sink(Mfull,query_hist,db_hists,eps,reg,tol,verbose=False):

    # -- constants --
    max_iters = 100

    # -- remove locations by non-zero --
    nz = torch.where( query_hist > 0 )[0]
    nz = query_hist.nonzero(as_tuple=True)[0]
    query_hist = query_hist[nz]
    M = Mfull[nz,:]

    # -- add eps for numerical stability --
    query_hist = query_hist# + eps
    db_hists = db_hists# + eps

    # -- alg 1 --
    K = torch.exp(-reg * M)
    Q = len(query_hist)
    N,D = db_hists.shape
    u = torch.ones( ( Q, N), device='cuda:0' ) / Q
    qdiag = torch.diag(1./query_hist)
    K_tilde = torch.mm(qdiag,K)
    delta,iters = 1,0
    while (delta > tol) and (iters < max_iters):
        u_old = u.clone()
        u = 1. / ( K_tilde @ ( db_hists.T / (K.T @ u ) ) )
        delta = torch.sum(torch.abs(u - u_old)).item()
        if verbose: print("delta: %2.2e" % delta)
        iters += 1
    v = db_hists.T / ( K.T @ u )
    d = torch.sum( u * ( ( K * M ) @ v ), axis=0)
    return d

def mp_lightspeed_sink(proc_index,outIndex,outDists,K,M,query_hists,db_hists,eps,reg,tol):

    query_hist = query_hists[proc_index]

    D,H = db_hists.shape
    # qwindow = torch.arange(q-100,q+100)
    qwindow = torch.arange(0,D)
    db_hists_view = db_hists[qwindow]

    # -- compute distances --
    distances = l2_hists(M,query_hist,db_hists_view,eps,reg,tol)
    # distances = lightspeed_sink(M,query_hist,db_hists_view,eps,reg,tol)
    indices = torch.argsort(distances)[:K].long()
    outIndex[q,:] = indices #qwindow[indices]
    outDists[q,:] = distances[indices]
    

