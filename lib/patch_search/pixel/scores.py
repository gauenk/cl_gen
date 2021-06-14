
# -- python imports --
import math
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict
from itertools import chain, combinations

# -- pytorch imports --
import torch
import torch.nn.functional as F

# -- project imports --
from pyutils import torch_xcorr,create_combination,print_tensor_stats,save_image,create_subset_grids,create_subset_grids,create_subset_grids_fixed,ncr,sample_subset_grids
from layers.unet import UNet_n2n,UNet_small
from layers.ot_pytorch import sink_stabilized,pairwise_distances,dmat

# -- [local] project imports --
from ..utils import get_ref_block_index

def get_score_functions(names):
    scores = edict()
    for name in names:
        scores[name] = get_score_function(name)
    return scores

def get_score_function(name):
    if name == "ave":
        return ave_score
    elif name == "gaussian_ot":
        return gaussian_ot_score
    elif name == "emd":
        return emd_score        
    elif name == "powerset":
        return powerset_score
    elif name == "extrema":
        return extrema_score
    elif name == "smsubset":
        return smsubset_score
    elif name == "lgsubset":
        return lgsubset_score
    elif name == "lgsubset_v_indices":
        return lgsubset_v_indices_score
    elif name == "lgsubset_v_ref":
        return lgsubset_v_ref_score
    elif name == "powerset_v_indices":
        return powerset_v_indices_score
    elif name == "powerset_v_ref_score":
        return powerset_v_ref_score
    elif name == "pairwise":
        return pairwise_delta_score
    elif name == "refcmp":
        return refcmp_score
    elif name == "jackknife":
        return jackknife
    elif name == "bootstrapping":
        return bootstrapping
    elif name == "sim_trm":
        return sim_trm
    elif name == "ransac":
        return ransac
    elif name == "shapley":
        return shapley
    else:
        raise ValueError(f"Uknown score function [{name}]")

def bootstrapping(cfg,expanded):

    # -- setup vars --
    R,B,E,T,C,H,W = expanded.shape
    device = expanded.device
    samples = rearrange(expanded,'r b e t c h w -> t (r c h w) (b e)')
    samples = samples.contiguous() # speed up?
    t_ref = T//2
    ave = torch.mean(samples,dim=0)

    # -- compute ave diff between model and subsets --
    scores_t = torch.zeros(T,B*E,device=device)
    counts_t = torch.zeros(T,1,device=device)
    nbatches,batch_size = 100,100
    for batch_idx in range(nbatches):
        subsets = torch.LongTensor(sample_subset_grids(T,batch_size))
        for subset in subsets:
            counts_t[subset] += 1
            subset_pix = samples[subset]
            vprint("subset.shape",subset_pix.shape)
            subset_ave = torch.mean(subset_pix,dim=0)
            vprint("subset_ave.shape",subset_ave.shape)
            loss = torch.mean( (subset_ave - ave)**2, dim=0)
            vprint("loss.shape",loss.shape)
            scores_t[subset] += loss
    scores_t /= counts_t
    scores = torch.mean(scores_t,dim=0)
    scores_t = scores_t.T # (T,E) -> (E,T)
    vprint("scores.shape",scores.shape)

    # -- no cuda --
    scores = rearrange(scores.cpu(),'(b e) -> b e',b=B)
    scores_t = rearrange(scores_t.cpu(),'(b e) t -> b e t',b=B)

    # -- add back patchsize for compat --
    scores = repeat(scores,'b e -> r b e',r=R)
    scores_t = repeat(scores_t,'b e t -> r b e t',r=R)

    return scores,scores_t

def shapley(cfg,expanded):
    """
    R = different patches from same image
    B = different images
    E = differnet block regions around centered patch
    T = burst of frames along a batch dimension
    """
    # -- setup --
    R,B,E,T,C,H,W = expanded.shape
    device = expanded.device
    samples = rearrange(expanded,'r b e t c h w -> t (r c h w) (b e)')
    samples = samples.contiguous() # speed up?
    t_ref = T//2

    def compute_pair_ave(x,y):
        mom_1 = torch.mean((x - y)**2,dim=0)
        return mom_1

    def convert_subsets(subsets,t):
        filtered_subsets = []
        for subset in subsets:
            if t in subset:
                filtered_subsets.append(subset)
        return filtered_subsets

    def join_subset(subset,t):
        return np.r_[subset,[t]]

    # -- create subset grids --
    minSN,maxSN,max_num = 1,15,100000
    indices = np.arange(T)
    subsets = create_subset_grids(minSN,maxSN,indices,max_num)
    size = 15 # 12 got (a) win and (b) match

    # -- init loop --
    ave = torch.mean(samples,dim=0)
    scores_t = torch.zeros(B*E,T,device=device)
    for t in range(T):
        subsets_rm_t = convert_subsets(subsets,t)
        for subset_rm_t in subsets_rm_t:

            # -- create new subset --
            subset_with_t = join_subset(subset_rm_t,t)
            
            # -- grab images --
            ave_with_t = torch.mean(samples[subset_with_t],dim=0)
            ave_rm_t = torch.mean(samples[subset_rm_t],dim=0)

            # -- compute differences --
            v_with_t = compute_pair_ave(ave_with_t,ave)
            v_rm_t = compute_pair_ave(ave_rm_t,ave)
            v_diff = -(v_with_t - v_rm_t)

            # -- compute normalization --
            s = len(subsets_rm_t)
            Z = ncr(s,T)**(-1)

            # -- accumulate --
            scores_t[:,t] += Z * v_diff

    # -- compute mean over frames --
    scores = torch.mean(scores_t,dim=1)

    # -- no cuda --
    scores = rearrange(scores.cpu(),'(b e) -> b e',b=B)
    scores_t = rearrange(scores_t.cpu(),'(b e) t -> b e t',b=B)

    # -- add back patchsize for compat --
    scores = repeat(scores,'b e -> r b e',r=R)
    scores_t = repeat(scores_t,'b e t -> r b e t',r=R)

    return scores,scores_t

def ransac(cfg,expanded):
    """
    R = different patches from same image
    B = different images
    E = differnet block regions around centered patch
    T = burst of frames along a batch dimension
    """
    # print("-> Print Time <-","\n\n\n")
    R,B,E,T,C,H,W = expanded.shape
    device = expanded.device
    samples = rearrange(expanded,'r b e t c h w -> t (r c h w) (b e)')
    samples = samples.contiguous() # speed up?
    t_ref = T//2

    
    def compute_std(gt_std,n_tr,n_te):
        std = np.sqrt(gt_std**2/n_tr + gt_std**2/n_te)
        return std

    def compute_pair_ave(x,y):
        mom_1 = torch.mean(torch.abs(x - y),dim=0)
        # mom_1 = torch.mean(x - y,dim=0)**2
        return mom_1

    def compute_pair_mom(x,y,nx,ny):
        mom_1 = (torch.mean(x,dim=0) - torch.mean(y,dim=0))**2
        mom_2 = (ny*torch.std(x,dim=0)**2 - nx*torch.std(y,dim=0)**2)**2
        return mom_1 + mom_2

    def compute_gaussian_ot(dist,gt_std):
        ave = torch.mean(dist,dim=0)
        std = torch.std(dist,dim=0)
        ot_loss = ave**2 + (std**2 - gt_std**2)**2
        return ot_loss

    def compute_loss_oos(tr_points,te_points,gt_std):
        # est.shape, N BE D
        # tr_ave = compute_train_est(tr_points)
        n_tr,n_te = len(tr_points),len(te_points)
        tr_ave = torch.mean(tr_points,dim=0)                
        te_ave = torch.mean(te_points,dim=0)
        dist = tr_ave - te_ave
        std = compute_std(gt_std,n_tr,n_te)
        loss = compute_gaussian_ot(dist,gt_std)
        return loss

    def compute_loss_oos_combo(tr_points,samples,ot_std,subsets_idx):

        # -- shape --
        T,D,BE = samples.shape
        device = tr_points.device
        
        # -- compute "model" --
        n_tr = len(tr_points)
        tr_ave = torch.mean(tr_points,dim=0)

        # -- create subsets to improve signal --
        T = samples.shape[0]
        ave = torch.mean(samples,dim=0)

        # -- compute ave diff between model and subsets --
        scores_t = torch.zeros(T,BE,device=device)
        counts_t = torch.zeros(T,1,device=device)
        for subset_idx in subsets_idx:
            nsub = T-len(subset_idx)+1
            counts_t[subset_idx] += 1
            subset = samples[subset_idx]
            vprint("subset.shape",subset.shape)
            subset_ave = torch.mean(subset,dim=0)
            vprint("subset_ave.shape",subset_ave.shape)
            n_sub = len(subset_idx)
            # loss = compute_pair_ave(tr_ave, subset_ave)
            # loss += compute_pair_ave(tr_ave, ave)
            # loss += compute_pair_ave(subset_ave, ave)
            # loss /= 3
            loss = compute_pair_ave(subset_ave, ave)
            loss /= (3 * nsub)
            # loss = compute_pair_mom(tr_ave, subset_ave, n_tr, n_sub)
            # dist = tr_ave - subset_ave
            # print_tensor_stats("tr_ave",tr_ave)
            # print_tensor_stats("subset_ave",subset_ave)
            # print_tensor_stats("dist",dist)
            # loss = compute_gaussian_ot(dist,ot_std)
            vprint("loss.shape",loss.shape)
            scores_t[subset_idx] += loss
        # n_evals = len(subsets_idx)
        # scores_t /= n_evals
        scores_t /= counts_t
        vprint(scores_t.shape)
        scores = torch.mean(scores_t,dim=0)
        scores_t = scores_t.T # (T,E) -> (E,T)
        vprint("scores.shape",scores.shape)
        return scores,scores_t

    def compute_train_est(tr_points):
        est = repeat(torch.mean(tr_points,dim=0),'be d -> n be d',n=te_N)

    # -=-=-=-=-=-=-=-=-=-=-=-
    #
    # -->      ransac     <--
    #
    # -=-=-=-=-=-=-=-=-=-=-=-

    # -- create subset grids --
    minSN,maxSN = 10,14
    indices = np.arange(T)
    max_subset_size = 600
    # subsets_idx = create_subset_grids_fixed(maxSN,indices,max_subset_size)
    subsets_idx = create_subset_grids(minSN,maxSN,indices,max_subset_size)

    # -- basic ransac hyperparams --
    desired_prob = .90
    # error_thresh = 1e-5
    size = 15 # 12 got (a) win and (b) match

    # -- set noise --
    gt_std = cfg.noise_params['g']['stddev']/255.
    ot_std = compute_std(gt_std,size,maxSN)
    ps_scale = (C*H*W*R)**(1/4.)
    # error_thresh = np.sqrt(gt_std**2/size + gt_std**2/maxSN) * 1.4 / ps_scale
    error_thresh = 1.
    # print(error_thresh)
    # error_thresh = np.sqrt(gt_std**2/size + gt_std**2/maxSN) * 1.96 / ps_scale
    # error_thresh = np.sqrt(gt_std**2/size + gt_std**2/maxSN) * 2.35 / ps_scale
    # print(gt_std)
    # print(ot_std)

    # -- init "bests" --
    scores_t = torch.zeros(B*E,T,device=device)
    scores = torch.zeros(B*E,device=device)
    best_scores = torch.ones(B*E,device=device) * float('inf')
    best_scores_t = torch.ones(B*E,T,device=device) * float('inf')

    # -- loop params --
    best_model = torch.zeros(B*E,T).int()
    best_model[:,t_ref] = 1
    # iters,num_iters,max_iters = 0,1000,1000
    # iters,num_iters,max_iters = 0,500,1000
    iters,num_iters,max_iters = 0,1,1000
    while iters < num_iters and iters < max_iters:

        # -- randomly select points with t_ref in train --
        order = torch.randperm(T)
        i_ref = torch.argmin(torch.abs(order - t_ref))
        order[i_ref] = order[0]
        order[0] = t_ref

        # -- split data --
        tr_index,te_index = order[:size],order[size:]
        tr_points,tr_N = samples[tr_index],len(tr_index)
        te_points,te_N = samples[te_index],len(te_index)
        # print(tr_index,te_index,best_model[0])
        
        # -- compute model --
        # loss_oos = compute_loss_oos(tr_points,samples,gt_std)
        # scores,scores_t = compute_loss_oos_combo(tr_points,te_points,ot_std,subsets_idx)
        losses,losses_t = compute_loss_oos_combo(tr_points,samples,ot_std,subsets_idx)
        # print(losses_t[:3,:])
        # print(losses_t[98,:])

        # -- adaptive threshold --
        # losses_std = torch.std(losses_t,dim=1)
        # losses_ave = torch.mean(losses_t,dim=1)
        # error_thresh = losses_ave + 1.96 * losses_std
        # print(error_thresh.shape,losses_t.shape)
        # print(losses_t)
        # print(losses_t[98])
        # exit()

        # -- check if best model --
        outliers = losses_t > error_thresh
        n_outliers = torch.sum(outliers,dim=1)
        # scores_t = outliers.float()
        # scores = torch.mean(scores_t,dim=1).float()
        args = torch.where(outliers)
        scores_t = losses_t.clone()
        scores_t[args[0],args[1]] += 1./T

        # inliers = losses_t < error_thresh
        # print(inliers.shape)
        # n_inliers = torch.sum(inliers,dim=1)
        # args = torch.where(inliers)
        # scores_t[args[0],args[1]] = losses_t[args[0],args[1]]
        # print(args[0].shape,args[1].shape,losses_t[args[0],args[1]].shape)
        # scores = torch.mean(scores_t,dim=1) * (10*torch.std(losses_t,dim=1))
        scores = torch.mean(scores_t,dim=1)
        # scores = torch.max(losses_t,dim=1).values - torch.min(losses_t,dim=1).values
        # print(scores[98],torch.mean(losses_t[98,:]))
        # print(scores[:3])
        # exit()
        
        # if inlier_count > max_inlier_count:
        #     max_inlier_count = inlier_count
        #     best_model = tr_index

        """
        We should not update our "best score"
        based on the average of _all_ samples
        
        We should update our score based on
        some component of inlier v.s. outliers
        
        An example includes the average score 
        of just the inliers...
        """
        # -- save best model for each batch --
        args = torch.where(scores < best_scores)[0]
        # if torch.any((args - 98) == 0):
        #     print("HI\n\n\n\n\n")
        #     print(scores[98])
        #     print("HI\n\n\n\n\n")
        nargs = len(args)
        args_rep = repeat(args,'nargs -> nargs n',n=len(tr_index))
        tr_index_rep = repeat(tr_index,'n -> nargs n',nargs=nargs)
        te_index_rep = repeat(te_index,'n -> nargs n',nargs=nargs)
        if len(args) > 0:
            best_scores[args] = scores[args]
            best_scores_t[args] = scores_t[args]
            best_model[args] = 0
            best_model[args_rep,tr_index_rep] = 1

        # -- update outlier prob --
        # prob_outlier = 1 - inlier_count/T
        # num_iters = math.log(1 - desired_prob)/math.log(1 - (1 - prob_outlier)**tr_size)
        iters = iters + 1

    # -- no cuda --
    scores = rearrange(best_scores.cpu(),'(b e) -> b e',b=B)
    scores_t = rearrange(best_scores_t.cpu(),'(b e) t -> b e t',b=B)
    best_model = rearrange(best_model.cpu(),'(b e) t -> b e t',b=B)

    bgrid = torch.arange(B)
    args = torch.argmin(scores,dim=1)

    # print(error_thresh)
    # print(args.shape)
    # print("argmin",args)
    # tgt_index = 98
    # print(best_model[:,tgt_index])
    # print(scores[:,tgt_index])
    # print(scores[bgrid,args])
    # print(scores[:,tgt_index] <= scores[bgrid,args])
    # exit()

    # -- add back patchsize for compat --
    scores = repeat(scores,'b e -> r b e',r=R)
    scores_t = repeat(scores_t,'b e t -> r b e t',r=R)

    return scores,scores_t

def ave_consistency(cfg,expanded):
    """
    R = different patches from same image
    B = different images
    E = differnet block regions around centered patch
    T = burst of frames along a batch dimension
    """
    # print("-> Print Time <-","\n\n\n")
    R,B,E,T,C,H,W = expanded.shape
    # expanded = expanded[[0]]
    expanded = rearrange(expanded,'r b e t c h w -> (b e) t (r c h w)')
    ave = torch.mean(expanded,dim=1)
    scores_t = []
    diffs = []
    t_ref = T//2
    for t in range(T):
        expanded_no_t = torch.cat([expanded[:,:t],expanded[:,t+1:]],dim=1)
        leave_out_t = torch.mean(expanded_no_t,dim=1)
        diffs.append(leave_out_t)
        # diffs_t = F.mse_loss(ave,leave_out_t,reduction='none')
        # diffs_t = T*(ave - leave_out_t)
        # diffs_t = expanded[:,t_ref,:] - expanded[:,t,:]
        # diffs.append(diffs_t)
        # ave_t = torch.mean(diffs_t,dim=1)
        # ac_vec_t,ac_coeff_t = torch_xcorr(diffs_t)
        # corr_term = -torch.log(torch.abs(ac_vec_t[:,0]))/10.
        # scores_t.append( torch.abs(ave_t) )#+  corr_term)
    # scores_t = torch.stack(scores_t,dim=1)

    def compare_diffs(x,y):
        mom_1 = (torch.mean(x,dim=1) - torch.mean(y,dim=1))**2
        mom_2 = (torch.std(x,dim=1)**2 - torch.std(y,dim=1)**2)**2
        return (mom_1 + mom_2).cpu()

    def compare_diffs_ot(x,y):
        x = rearrange(x,'be (r c hw) -> be (r hw) c',r=R,c=3)
        y = rearrange(y,'be (r c hw) -> be (r hw) c',r=R,c=3)
        dists = []
        BE = x.shape[0]
        for be in range(BE):
            M_be = dmat(x[be],y[be])
            # M_be = pairwise_distances(x[be],y[be])
            M_be = M_be.to(x.device)
            dist = sink_stabilized(M_be,reg=1.0,device=x.device).item()
            dists.append(dist)
        return torch.FloatTensor(dists)

    def compare_to_known(x,gt_std):
        mom_1 = (torch.mean(x,dim=1) - 0)**2
        mom_2 = (torch.std(x,dim=1) - gt_std)**2
        # ac_vec_t,ac_coeff_t = torch_xcorr(diffs_t)
        # rev_x = torch.flip(x,(1,))
        # xcorr = torch.einsum('bi,bj->b',x,rev_x)
        # ac_coeff_t = torch.mean(torch.abs(ac_vec_t[:,1:2]),dim=1)
        # print(ac_vec_t[:,0])
        # return (mom_1 + mom_2).cpu()# + ac_coeff_t).cpu()
        return (mom_1 + mom_2 + xcorr).cpu()# + ac_coeff_t).cpu()
        # return (mom_1 + mom_2 + ac_coeff_t).cpu()

    # print(ave.shape)
    # print(torch.mean(ave[0]),torch.std(ave[0]))
    # print(torch.mean(ave[1]),torch.std(ave[1]))

    cmps = []
    cmps = {str(t):0 for t in range(T)}
    # scores_t = torch.zeros((R*B*E,T))
    scores_t = torch.zeros((B*E,T))
    for t_i in range(T):
        for t_j in range(T):
            if t_i >= t_j: continue
            #comp_t_ij = compare_diffs(diffs[t_i],diffs[t_j])
            comp_t_ij = torch.sum(torch.abs(diffs[t_i]-diffs[t_j]),dim=1).cpu()
            scores_t[:,t_i] += comp_t_ij
            scores_t[:,t_j] += comp_t_ij
            # cmps[str(t_i)] += 1
            # cmps[str(t_j)] += 1
            # cmps.append(comp_t_ij)
    scores = torch.mean(scores_t,dim=1)
    scores_t = repeat(rearrange(scores_t,'(b e) t -> b e t',b=B,e=E),
                      'b e t -> r b e t',r=R)
    scores = repeat(rearrange(scores,'(b e) -> b e',b=B,e=E),
                    'b e -> r b e',r=R)
    # -- to cpu --
    scores = scores.cpu()
    scores_t = scores_t.cpu()

    return scores,scores_t

    
def vprint(*args):
    verbose = False
    if verbose:
        print(*args)

def sim_trm(cfg,expanded):
    vprint("-> Print Time <-","\n\n\n")
    R,B,E,T,C,H,W = expanded.shape
    # expanded = expanded[[0]]
    expanded = rearrange(expanded,'r b e t c h w -> (b e) t (r c h w)')
    ave = torch.mean(expanded,dim=1)
    scores_t = []
    diffs = []
    t_ref = T//2
    for t in range(T):
        expanded_no_t = torch.cat([expanded[:,:t],expanded[:,t+1:]],dim=1)
        leave_out_t = torch.mean(expanded_no_t,dim=1)
        # diffs.append(leave_out_t)
        # diffs_t = F.mse_loss(ave,leave_out_t,reduction='none')
        # diffs_t = T*(ave - leave_out_t)
        # diffs_t = expanded[:,t_ref,:] - expanded[:,t,:]
        # diffs.append(diffs_t)
        # scores_t.append( torch.abs(ave_t) )#+  corr_term)
        scores_t.append( torch.mean(leave_out_t - 0.5,dim=1).cpu() )#+  corr_term)
    scores_t = torch.stack(scores_t,dim=1)

    ave = rearrange(ave,'(b e) d -> b e d',b=B)
    scores = torch.zeros((B,E))
    for e_i in range(E):
        for e_j in range(E):
            if e_i >= e_j: continue
            l2_diff = torch.mean((ave[:,e_i] - ave[:,e_j])**2,dim=1).cpu()
            scores[:,e_i] += l2_diff
            scores[:,e_j] += l2_diff

    # scores = torch.mean(scores_t,dim=1)

    # -- be -> b e --
    scores_t = rearrange(scores_t,'(b e) t -> b e t',b=B,e=E)
    # scores = rearrange(scores,'(b e) -> b e',b=B,e=E)

    # -- repeat --
    scores_t = repeat(scores_t,'b e t -> r b e t',r=R)
    scores = repeat(scores,'b e -> r b e',r=R)

    # -- to cpu --
    scores = scores.cpu()
    scores_t = scores_t.cpu()
    print(torch.argmin(scores[0],1))

    return scores,scores_t

def jackknife(cfg,expanded):
    """
    R = different patches from same image
    B = different images
    E = differnet block regions around centered patch
    T = burst of frames along a batch dimension
    """
    vprint("-> Print Time <-","\n\n\n")
    R,B,E,T,C,H,W = expanded.shape
    # expanded = expanded[[0]]
    expanded = rearrange(expanded,'r b e t c h w -> (b e) t (r c h w)')
    ave = torch.mean(expanded,dim=1)
    scores_t = []
    diffs = []
    t_ref = T//2
    for t in range(T):
        expanded_no_t = torch.cat([expanded[:,:t],expanded[:,t+1:]],dim=1)
        leave_out_t = torch.mean(expanded_no_t,dim=1)
        # diffs.append(leave_out_t)
        # diffs_t = F.mse_loss(ave,leave_out_t,reduction='none')
        diffs_t = T*(ave - leave_out_t)
        # diffs_t = expanded[:,t_ref,:] - expanded[:,t,:]
        diffs.append(diffs_t)
        # ave_t = torch.mean(diffs_t,dim=1)
        # ac_vec_t,ac_coeff_t = torch_xcorr(diffs_t)
        # corr_term = -torch.log(torch.abs(ac_vec_t[:,0]))/10.
        # scores_t.append( torch.abs(ave_t) )#+  corr_term)
        # scores_t.append( torch.mean(leave_out_t - 0.5,dim=1).cpu() )#+  corr_term)
    # scores_t = torch.stack(scores_t,dim=1)

    def compare_diffs(x,y):
        mom_1 = (torch.mean(x,dim=1) - torch.mean(y,dim=1))**2
        mom_2 = (torch.std(x,dim=1)**2 - torch.std(y,dim=1)**2)**2
        return (mom_1 + mom_2).cpu()

    def compare_diffs_ot(x,y):
        x = rearrange(x,'be (r c hw) -> be (r hw) c',r=R,c=3)
        y = rearrange(y,'be (r c hw) -> be (r hw) c',r=R,c=3)
        dists = []
        BE = x.shape[0]
        for be in range(BE):
            M_be = dmat(x[be],y[be])
            # M_be = pairwise_distances(x[be],y[be])
            M_be = M_be.to(x.device)
            dist = sink_stabilized(M_be,reg=1.0,device=x.device).item()
            dists.append(dist)
        return torch.FloatTensor(dists)

    def compare_to_known(x,gt_std):
        mom_1 = (torch.mean(x,dim=1) - 0)**2
        mom_2 = (torch.std(x,dim=1) - gt_std)**2
        # ac_vec_t,ac_coeff_t = torch_xcorr(diffs_t)
        # rev_x = torch.flip(x,(1,))
        # xcorr = torch.einsum('bi,bj->b',x,rev_x)
        # ac_coeff_t = torch.mean(torch.abs(ac_vec_t[:,1:2]),dim=1)
        # print(ac_vec_t[:,0])
        return (mom_1 + mom_2).cpu()# + ac_coeff_t).cpu()
        # return (mom_1 + mom_2 + xcorr).cpu()# + ac_coeff_t).cpu()
        # return (mom_1 + mom_2 + ac_coeff_t).cpu()

    # print(ave.shape)
    # print(torch.mean(ave[0]),torch.std(ave[0]))
    # print(torch.mean(ave[1]),torch.std(ave[1]))

    # cmps = []
    # cmps = {str(t):0 for t in range(T)}
    # # scores_t = torch.zeros((R*B*E,T))
    # scores_t = torch.zeros((B*E,T))
    # for t_i in range(T):
    #     for t_j in range(T):
    #         if t_i >= t_j: continue
    #         #comp_t_ij = compare_diffs(diffs[t_i],diffs[t_j])
    #         comp_t_ij = torch.sum(torch.abs(diffs[t_i]-diffs[t_j]),dim=1).cpu()
    #         scores_t[:,t_i] += comp_t_ij
    #         scores_t[:,t_j] += comp_t_ij
    #         # cmps[str(t_i)] += 1
    #         # cmps[str(t_j)] += 1
    #         # cmps.append(comp_t_ij)

    t_ref = T//2
    noise = cfg.noise_params['g']['stddev']/255.
    gt_std = np.sqrt((noise**2 + noise**2/(T-1)))
    # scores_t = torch.zeros((R*B*E,T))
    # vprint(ave.shape)
    # for i in range(10):
    #     vprint('ave',torch.mean(ave,dim=1)[i],torch.std(ave,dim=1)[i],gt_std)
    #     for t in range(T):
    #         vprint(t,torch.mean(diffs[t],dim=1)[i],torch.std(diffs[t],dim=1)[i],gt_std)

    scores_t = torch.zeros((B*E,T))
    for t in range(T):
        #print(t,torch.mean(diffs[t],dim=1)[0],torch.std(diffs[t],dim=1)[0],gt_std)
        # comp_t = compare_diffs(diffs[t],diffs[t_ref])
        # comp_t = compare_diffs_ot(diffs[t],diffs[t_ref])
        comp_t = compare_to_known(diffs[t],gt_std)
        # comp_t = torch.mean((diffs[t] - diffs[t_ref])**2,dim=1).cpu()
        scores_t[:,t] += comp_t

    # print(cmps)
    # print(scores_t)
    scores = torch.mean(scores_t,dim=1)

    scores_t = repeat(rearrange(scores_t,'(b e) t -> b e t',b=B,e=E),
                      'b e t -> r b e t',r=R)
    scores = repeat(rearrange(scores,'(b e) -> b e',b=B,e=E),
                    'b e -> r b e',r=R)

    # -- to cpu --
    scores = scores.cpu()
    scores_t = scores_t.cpu()

    # scores_t = rearrange(scores_t,'(r b e) t -> r b e t',r=R,b=B,e=E)
    # scores = rearrange(scores,'(r b e) -> r b e',r=R,b=B,e=E)
    # print(scores.shape)
    # print(torch.argmin(scores[0],1))

    return scores,scores_t
    
def ave_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape
    """
    R = different patches from same image
    B = different images
    E = differnet block regions around centered patch
    T = burst of frames along a batch dimension
    """
    # -- R goes to CHW --
    expanded = rearrange(expanded,'r b e t c h w -> t b e (r c h w)')
    
    ref = repeat(expanded[T//2],'b e d -> tile b e d',tile=T-1)
    neighbors = torch.cat([expanded[:T//2],expanded[T//2+1:]],dim=0)
    delta = F.mse_loss(ref,neighbors,reduction='none')
    delta_t = torch.mean(delta,dim=3)
    delta = torch.mean(delta_t,dim=0)

    # -- append dim for T --
    Tm1 = T-1
    delta_t = rearrange(delta_t,'t b e -> b e t')
    zeros = torch.zeros_like(delta_t[:,:,[0]])
    delta_t = torch.cat([delta_t[:,:,:Tm1//2],zeros,delta_t[:,:,Tm1//2:]],dim=2)

    # -- repeat to include R --
    delta_t = repeat(delta_t,'b e t -> r b e t',r=R)
    delta = repeat(delta,'b e -> r b e',r=R)

    return delta,delta_t


def ave_score_original(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape
    """
    R = different patches from same image
    B = different images
    E = differnet block regions around centered patch
    T = burst of frames along a batch dimension
    """
    ref = repeat(expanded[:,:,:,T//2],'r b e c h w -> r b e tile c h w',tile=T-1)
    neighbors = torch.cat([expanded[:,:,:,:T//2],expanded[:,:,:,T//2+1:]],dim=3)
    delta = F.mse_loss(ref,neighbors,reduction='none')
    delta = delta.view(R,B,E,T-1,-1)
    delta_t = torch.mean(delta,dim=4)
    delta = torch.mean(delta_t,dim=3)

    # -- append dim for T --
    Tm1 = T-1
    zeros = torch.zeros_like(delta_t[:,:,:,[0]])
    delta_t = torch.cat([delta_t[:,:,:,:Tm1//2],zeros,delta_t[:,:,:,Tm1//2:]],dim=3)

    return delta,delta_t

def refcmp_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape
    ref = expanded[:,:,:,T//2]
    neighbors = torch.cat([expanded[:,:,:,:T//2],expanded[:,:,:,T//2+1:]],dim=3)
    delta_t = torch.zeros(R,B,E,T,device=expanded.device)
    delta = torch.zeros(R,B,E,device=expanded.device)
    for t in range(T-1):
        delta_pair = F.mse_loss(neighbors[:,:,:,t],ref,reduction='none')
        delta_t += delta_pair
        delta += torch.mean(delta_t.view(R,B,E,-1),dim=3)
    delta_t /= (T-1)
    delta /= (T-1)
    return delta,delta_t

def pairwise_delta_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape
    # ref = repeat(expanded[:,:,:,[T//2]],'r b e c h w -> r b e tile c h w',tile=T-1)
    # neighbors = torch.cat([expanded[:,:,:,:T//2],expanded[:,:,:,T//2+1:]],dim=3)
    delta_t = torch.zeros(R,B,E,T,device=expanded.device)
    delta = torch.zeros(R,B,E,device=expanded.device)
    for t1 in range(T):
        for t2 in range(T):
            delta_pair = F.mse_loss(expanded[:,:,:,t1],expanded[:,:,:,t2],reduction='none')
            delta_t[:,:,:,[t1,t2]] += delta_pair
            delta += torch.mean(delta_t.view(R,B,E,-1),dim=3)
    delta /= T*T
    delta_t /= T*T
    return delta,delta_t

#
# Grid Functions
#

# -- run over the grids for below --
def delta_over_grids(cfg,expanded,grids):
    R,B,E,T,C,H,W = expanded.shape
    unrolled = rearrange(expanded,'r b e t c h w -> r b e t (c h w)')
    delta_t = torch.zeros(R,B,E,T,device=expanded.device)
    delta = torch.zeros(R,B,E,device=expanded.device)
    for set0,set1 in grids:
        set0,set1 = np.atleast_1d(set0),np.atleast_1d(set1)

        # -- compute ave --
        ave0 = torch.mean(expanded[:,:,:,set0],dim=3)
        ave1 = torch.mean(expanded[:,:,:,set1],dim=3)

        # -- rearrange --
        ave0 = rearrange(ave0,'r b e c h w -> r b e (c h w)')
        ave1 = rearrange(ave1,'r b e c h w -> r b e (c h w)')

        # -- rep across time --
        ave0_repT = repeat(ave0,'r b e f -> r b e t f',t=T)
        ave1_repT = repeat(ave1,'r b e f -> r b e t f',t=T)

        # -- compute deltas --
        delta_pair = F.mse_loss(ave0,ave1,reduction='none').view(R,B,E,-1)
        delta_0 = F.mse_loss(ave0_repT,unrolled,reduction='none').view(R,B,E,T,-1)
        delta_1 = F.mse_loss(ave1_repT,unrolled,reduction='none').view(R,B,E,T,-1)
        delta_t += torch.mean( (delta_0 + delta_1)/2., dim = 4)
        delta += torch.mean(delta_pair,dim=3)
    delta /= len(grids)
    delta_t /= len(grids)
    return delta,delta_t

def powerset_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- powerset --
    indices = np.arange(T)
    powerset = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(T))
    powerset = np.array([np.array(elem) for elem in list(powerset)])
    grids = np.array(np.meshgrid(*[powerset,powerset]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

def extrema_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- extrema subsets --
    indices = np.arange(T)
    subset_lg = create_combination(indices,T-2,T)
    subset_sm = create_combination(indices,0,2)
    subset_ex = np.r_[subset_sm,subset_lg]
    grids = np.array(np.meshgrid(*[subset_ex,subset_ex]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

def smsubset_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- compare large subsets --
    indices = np.arange(T)
    subset_sm = create_combination(indices,1,2)
    grids = np.array(np.meshgrid(*[subset_sm,subset_sm]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

def lgsubset_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- compare large subsets --
    indices = np.arange(T)
    subset_lg = create_combination(indices,T-2,T)
    grids = np.array(np.meshgrid(*[subset_lg,subset_lg]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t


def lgsubset_v_indices_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- indices and large subset --
    indices = np.arange(T)
    l_indices = [[i] for i in indices]
    l_indices.append(list(np.arange(T)))
    subset_lg = create_combination(indices,T-2,T)
    grids = np.array(np.meshgrid(*[subset_lg,l_indices]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

def lgsubset_v_ref_score(cfg,expanded,ref_t=None):
    R,B,E,T,C,H,W = expanded.shape
    if ref_t is None: ref_t = T//2

    # -- indices and large subset --
    indices = np.arange(T)
    subset_lg = create_combination(indices,T-2,T)
    grids = np.array(np.meshgrid(*[subset_lg,[[ref_t,]]]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

def powerset_v_indices_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- indices and large subset --
    indices = np.arange(T)
    l_indices = [[i] for i in indices]
    l_indices.append(list(np.arange(T)))
    powerset = create_combination(indices,0,T)
    grids = np.array(np.meshgrid(*[powerset,l_indices]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

def powerset_v_ref_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- indices and large subset --
    indices = np.arange(T)
    powerset = create_combination(indices,0,T)
    grids = np.array(np.meshgrid(*[powerset,[T//2]]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

#
# Optimal Transport Based Losses
# 

def gaussian_ot_score(cfg,expanded,return_frames=False):
    R,B,E,T,C,H,W = expanded.shape
    vectorize = rearrange(expanded,'r b e t c h w -> (r b e t) (c h w)')
    means = torch.mean(vectorize,dim=1)
    stds = torch.std(vectorize,dim=1)

    # -- gaussian zero mean, var = noise_level --
    gt_std = cfg.noise_params['g']['stddev']/255.
    loss = means**2
    loss += (stds**2 - 2*gt_std**2)**2
    losses_t = rearrange(loss,'(r b e t) -> r b e t',r=R,b=B,e=E,t=T)
    losses = torch.mean(losses_t,dim=3)
    return losses,losses_t

def emd_score(cfg,expanded):
    pass

