"""
How fast can we train a unet to denoise a the same
image content with different noise patterns?
"""

# -- python imports --
import sys,os,random,re
from tqdm import tqdm
import pandas as pd
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from easydict import EasyDict as edict
from einops import rearrange,repeat

# -- pytorch import --
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as tv_utils
import torchvision.transforms.functional as tvF
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

# -- project code --
import settings
from pyutils.timer import Timer
from datasets import load_dataset
from datasets.transforms import get_noise_transform,get_dynamic_transform
from pyutils import np_log,rescale_noisy_image,mse_to_psnr,count_parameters,images_to_psnrs
from pyutils.plot import add_legend
from layers.unet import UNet_n2n,UNet_small
from patch_search import get_score_function
from patch_search.cog import COG,Logistic,AlignmentDetector,score_cog
from .trace import activation_trace
from .utils import get_block_arangements,get_ref_block_index,save_image,get_block_arangements_subset,get_small_test_block_arangements,print_tensor_stats,sample_good_init_tl,crop_burst_to_blocks
from .explore_fast_unet import train,test

COG_DIR = Path(f"{settings.ROOT_PATH}/output/lpas/cog/")
if not COG_DIR.exists(): COG_DIR.mkdir()

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def ddp_cleanup():
    pass

def fill_results(cfg,image,clean,burst,model,idx):

    results = {}
    # -- fill in old result params --
    T = burst.shape[0]
    rep = repeat(image,'c h w -> tile c h w',tile=T)
    psnr_clean = float(np.mean(images_to_psnrs(rep,clean)))
    results['psnr_clean'] = psnr_clean

    results['params_norm_mean'] = -1
    results['trace_norm'] = -1
    results['mse'] = -1
    results['psnr_rec'] = -1
    results['mse'] = -1
    results['psnr_burst'] = -1
    results['psnr_intra_input'] = -1
    results['psnr_bc_v1'] = -1
    results['psnr_noisy'] = -1
    score_fxn_names = ['lgsubset_v_ref','lgsubset','ave','lgsubset_v_indices','gaussian_ot']
    for name in score_fxn_names: results[f"fu_{name}"] = 0.
    for name in score_fxn_names: results[name] = 0.
    return results

def score_function_wrapper(score_fxn):
    def wrapper(cfg,image):
        tmp = image.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        scores = score_fxn(cfg,tmp)
        return scores[0,0,0]
    return wrapper


def zo_nmlz(ndarray,eps=1e-13):
    ndarray -= ndarray.min()
    ndarray /= (ndarray.max() + eps)
    return ndarray

def extrema_scores(scores,K):
    order = np.argsort(scores)
    topK = edict({'grid':[],'values':[]})
    bottomK = edict({'grid':[],'values':[]})
    extrema = edict({'grid':[],'values':[]})
    topK.grid = order[:K]
    topK.values = scores[order[:K]]
    bottomK.grid = order[-K:]
    bottomK.values = scores[order[-K:]]
    extrema.grid = np.r_[topK.grid,bottomK.grid]
    extrema.values = np.r_[topK.values,bottomK.values]
    return extrema,topK,bottomK
    
def jitter(ndarray,std=1e-2):
    return npr.normal(ndarray,std)

def explore_cog_record(cfg,record,bss_dir=None):
    REF_H = get_ref_block_index(cfg.nblocks)
    # cfg.nblocks,cfg.nframes = 5,7

    # -- load block search space --
    tcount = 3
    size = 30
    bss = get_small_test_block_arangements(bss_dir,cfg.nblocks,cfg.nframes,tcount,
                                           size,difficult=True)
    block_search_space = bss 
    # if block_search_space_fn:
    #     block_search_space = np.load(block_search_space_fn,allow_pickle=True)
    # else:
    #     block_search_space = get_block_arangements(cfg.nblocks,cfg.nframes)

    # -- print un-modified results --
    for ridx,field in enumerate(record.columns):
        if field == "Unnamed: 0": continue
        # -- printing best index and score --
        index = np.argmin(record[field])
        search = block_search_space[index]
        if torch.is_tensor(search): search = search.numpy()
        max_index = np.argmax(record[field])
        max_search = block_search_space[max_index]
        max_value = record[field][max_index]

        min_index = np.argmin(record[field])
        min_search = block_search_space[min_index]
        min_value = record[field][min_index]

        print(field)
        print("\t",max_search,max_index,max_value)
        print("\t",min_search,min_index,min_value)


    # -- prepare data for plotting --
    P = len(record)
    pgrid = np.arange(P)
    psnr_clean = jitter(zo_nmlz(record['psnr_clean'].to_numpy()))
    psnr_noisy = jitter(zo_nmlz(record['psnr_noisy'].to_numpy()))
    psnr_rec = jitter(zo_nmlz(record['psnr_rec'].to_numpy()))
    cog = jitter(zo_nmlz(record['cog'].to_numpy()))
    ave = jitter(zo_nmlz(record['ave'].to_numpy()))

    # -- plot --
    naxs,K = 2,3
    rm_fields = ["psnr_clean","cog","ave","psnr_noisy","psnr_rec"]
    ncols = len(record.columns) - len(rm_fields)
    fig,ax = plt.subplots(naxs,1,figsize=(12,10))
    plt.subplots_adjust(hspace=0.3)
    ax[0].plot(pgrid,psnr_clean,'x',label='clean')
    ax[0].plot(pgrid,psnr_noisy,'x',label='noisy')
    ax[0].plot(pgrid,psnr_rec,'+',label='rec')
    ax[0].plot(pgrid,cog,'+',label='cog')
    ax[0].plot(pgrid,ave,'+',label='ave')
    add_legend(ax[0],"cmp",['clean','noisy','rec','cog','ave'])
    idx,ax_idx,ax_mod = 0,1,100#ncols//(naxs-1)+1
    labels = [[] for _ in range(naxs-1)]
    print(ax_mod)
    for ridx,field in enumerate(record.columns):

        # -- plotting search type --
        if field in rm_fields: continue
        search = re.match("[0-9]f",field[-2:])
        if search is not None: continue

        scores = record[field].to_numpy()
        if not isinstance(scores[0],float): continue
        scores = jitter(zo_nmlz(scores))
        # ax[ax_idx].plot(pgrid,scores,'x-',alpha=0.8)
        extrema,topK,bottomK = extrema_scores(scores,K)
        print(idx)
        if idx < 8: mrk = 'x'
        else: mrk = '+'
        if "fu_" in field or "fnet_" in field: mrk += '-'
        vorder = np.argsort(extrema.values)
        print(np.abs(extrema.values[vorder[0]] - extrema.values[vorder[1]]))
        iorder = np.argsort(extrema.grid)
        grid = extrema.grid[iorder]
        values = extrema.values[iorder]
        
        ax[ax_idx].plot(grid,values,mrk,alpha=0.8,label=field)
        if "fu_" in field:
            lfield = field.replace("fu_","fnet_")
        else: lfield = field
        labels[ax_idx-1].append(lfield)
        idx += 1
        if idx % ax_mod == 0:
            idx = 0
            ax_idx += 1
            print(idx,ax_idx,ax_mod)
    for i in range(naxs-1):
        add_legend(ax[i+1],"cmp",labels[i])

    plot_dir = Path("./output/lpas/cog/")
    if not plot_dir.exists(): plot_dir.mkdir()
    plt.savefig(plot_dir / Path("cog_psnrs_extrema.png"))
    

def run_experiment(cfg,data,record_fn,bss_dir):

    #
    # Experiment Setup
    # 

    # -- init variables for access --
    T = cfg.nframes
    H = cfg.nblocks
    framesize = 156
    patchsize = 32
    P = 9
    REF_H = get_ref_block_index(cfg.nblocks)
    nn_params = edict({'lr':1e-3,'init_params':None})
    gridT = torch.arange(T)

    # -- setup noise --
    cfg.noise_type = 'g'
    cfg.ntype = cfg.noise_type
    cfg.noise_params.ntype = cfg.noise_type
    noise_level = 25.
    cfg.noise_params['g']['stddev'] = noise_level
    noise_level_str = f"{int(noise_level)}"
    noise_xform = get_noise_transform(cfg.noise_params,use_to_tensor=False)

    # -- simulate no motion --
    nomotion = np.zeros((T,2)).astype(np.long)
    image_index = 10
    full_image = data.tr[image_index][2]
    full_burst,aligned,motion = simulate_dynamics(full_image,T,nomotion,0,framesize)
    save_image(full_burst,"full_burst.png")
    clean_full_ref = full_burst[T//2]

    # -- apply noisy --
    full_noisy = noise_xform(full_burst)
    save_image(full_noisy,"full_noisy.png")
    noisy_full_ref = full_noisy[T//2]

    #
    # Start Method
    # 

    # -- find good patches from full noisy image --
    # init_tl_list = sample_good_init_tl(clean_full_ref,P,patchsize)
    init_tl_list = sample_good_init_tl(noisy_full_ref,P,patchsize)

    # -- grab blocks from selected patches for burst --
    clean,noisy = [],[]
    clean = crop_burst_to_blocks(full_burst,cfg.nblocks,init_tl_list[0],patchsize)
    noisy = crop_burst_to_blocks(full_noisy,cfg.nblocks,init_tl_list[0],patchsize)


    # -- image for "test" function --
    REF_PATCH = 0 # backward compat. for functions without patch-dim support
    # image = clean[T//2,REF_H,REF_PATCH] # legacy
    image = noisy[T//2,REF_H,REF_PATCH]+0.5 # legacy    
    
    # -- normalize --
    # clean -= clean.min()
    # clean /= clean.max()
    print_tensor_stats("clean",clean)
    print_tensor_stats("noisy",clean)
    print_tensor_stats("full_burst",full_burst)
    print_tensor_stats("full_noisy",full_noisy)

    save_image(clean,"fast_unet_clean.png",normalize=True)
    save_image(noisy,"fast_unet_noisy.png",normalize=True)
    save_image(image,"fast_unet_image.png",normalize=True)

    # -- select search space --
    # block_search_space = get_block_arangements_subset(cfg.nblocks,cfg.nframes,
    #                                                   tcount=4,difficult=True)
    # block_search_space = get_block_arangements(cfg.nblocks,cfg.nframes)
    use_rand = True
    if use_rand:
        tcount = 3
        size = 30
        bss = get_small_test_block_arangements(bss_dir,cfg.nblocks,cfg.nframes,
                                               tcount,size,difficult=True)
        print("LEN BSS",len(bss))
        block_search_space = bss 
    
    # -- setup loop --
    clean = clean.to(cfg.device)
    noisy = noisy.to(cfg.device)
    image = image.to(cfg.device)
    record,idx = [],0

    # -- search over search space --
    for prop in tqdm(block_search_space):

        # -- fetch --
        clean_prop = clean[gridT,prop].to(cfg.device)
        noisy_prop = noisy[gridT,prop].to(cfg.device)
        save_image(clean_prop[:,0],"clean_prop.png")
        save_image(noisy_prop[:,0],"noisy_prop.png")

        # -- compute COG --
        backbone = UNet_small
        if (nn_params['init_params'] is None): train_steps = cfg.cog_train_steps
        else: train_steps = cfg.cog_second_train_steps
        score = 0.
        # score = score_cog(cfg,image_volume,backbone,nn_params,train_steps)

        # -- [legacy] fill results with -1's --
        # results = fill_results(cfg,image,clean_prop,noisy_prop,None,idx)

        # -- compute single UNet --
        model = UNet_small(3).to(cfg.device)
        init_lr = 1e-4
        optim = torch.optim.Adam(model.parameters(),lr=init_lr,betas=(0.9,0.99))
        train(cfg,image,clean_prop[:,REF_PATCH],noisy_prop[:,REF_PATCH],model,optim)
        print("noisy_prop.shape",noisy_prop.shape)
        results = test(cfg,image,clean_prop[:,REF_PATCH],noisy_prop[:,REF_PATCH],model,idx)
        results['cog'] = score
        print(score,results['ave'],results['psnr_clean'],prop)

        # -- update --
        record.append(results)
        idx += 1

    record = pd.DataFrame(record)
    print(f"Writing record to {record_fn}")
    record.to_csv(record_fn)
    return record

def iter_over_image_unet(cfg,data):
    full_image = data.tr[0][2]
    queue = None
    acc = single_image_unet(cfg,queue,full_image,'cuda:0')
    print(f"[Run 0]:",acc)

def mp_iter_over_image_unet(cfg,data):

    running_acc,processes = 0,[]
    mp.set_start_method('spawn')
    nprocs,ngpus = 3,3
    queue = mp.Queue()
    success = []
    for i in range(45):
        full_image = data.tr[i][2]
        device = f'cuda:{i % ngpus}'
        p = mp.Process(target=single_image_unet,args=(cfg,queue,full_image,device))
        p.start()
        processes.append(p)
        if len(processes) >= nprocs:
            for proc in processes:
                results = queue.get()
                succ,acc = results[0],results[1]
                success.append(succ)
                proc.join()
                running_acc += acc
            print( f"[Run %d]: %2.3f | %2.3f" % (i, acc, running_acc/(i+1)) )
            npsuccess = np.array(success)
            print(np.mean(npsuccess),np.std(npsuccess))
            print(np.mean(npsuccess,axis=0),np.std(npsuccess,axis=0))
            print(np.mean(npsuccess,axis=1),np.std(npsuccess,axis=1))

            processes = []
    npsuccess = np.array(success)
    print(np.mean(npsuccess),np.std(npsuccess))
    print(np.mean(npsuccess,axis=0),np.std(npsuccess,axis=0))
    print(np.mean(npsuccess,axis=1),np.std(npsuccess,axis=1))

def single_image_unet(cfg,queue,full_image,device):
    full_image = full_image.to(device)
    image = tvF.crop(full_image,128,128,32,32)
    T = 5

    # -- poisson noise --
    noise_type = "pn"
    cfg.noise_type = noise_type
    cfg.noise_params['pn']['alpha'] = 40.0
    cfg.noise_params['pn']['readout'] = 0.0
    cfg.noise_params.ntype = cfg.noise_type
    noise_xform = get_noise_transform(cfg.noise_params,use_to_tensor=False)

    clean = torch.stack([image for i in range(T)],dim=0)
    noisy = noise_xform(clean)
    save_image(clean,"clean.png",normalize=True)
    m_clean,m_noisy = clean.clone(),noisy.clone()
    for i in range(T//2):
        image_mis = tvF.crop(full_image,128+1,128,32,32)
        m_clean[i] = image_mis
        m_noisy[i] = noise_xform(m_clean[i])
    # m_clean[1] = image_mis
    # m_noisy[1] = torch.normal(m_clean[0]-0.5,noise_level)

    save_image(clean,"clean.png")
    save_image(noisy,"noisy.png")
    save_image(m_clean,"mis_clean.png")
    save_image(m_noisy,"m_noisy.png")

    lacc = []
    acc,nruns = 0,1
    train_steps = 3000
    for i in range(nruns):

        # -- create COG --
        print("aligned")
        cog = COG(UNet_small,T,image.device,nn_params=None,train_steps=train_steps)
        cog.train_models(noisy)
        recs = cog.test_models(noisy)
        score_align = cog.operator_consistency(recs,noisy)
        save_image(noisy,"noisy.png")
        save_image(recs,"recs_aligned.png")
        # noisy_rep = repeat(noisy,'t c h w -> tile t c h w',tile=T)
        # score_align = cog.compute_consistency(recs,noisy_rep)
        
        print("misaligned")
        cog = COG(UNet_small,T,image.device,nn_params=None,train_steps=train_steps)
        cog.train_models(m_noisy)
        recs = cog.test_models(m_noisy)
        save_image(m_noisy,"m_noisy.png")
        save_image(recs,"recs_misaligned.png")
        score_misalign = cog.operator_consistency(recs,m_noisy)
        # print("misaligned")
        # m_noisy_rep = repeat(m_noisy,'t c h w -> tile t c h w',tile=T)
        # score_misalign = cog.compute_consistency(recs,m_noisy_rep)

        # print(f"[Run {i}]:",score_align,score_misalign)
        lacc.append((score_align > score_misalign))
        acc += score_align > score_misalign

    # print("Acc: %2.3f" % (100*(acc / nruns)) )
    acc = acc / float(nruns)
    if not (queue is None): queue.put((lacc,acc))
    # print(F.mse_loss(m_clean[0][:,32//2,32//2],clean[0][:,32//2,32//2]).item())
    return acc


def sim_posneg_one(can_zero=False):
    if can_zero:
        x = torch.randint(0,3,(1,))[0].item()
        return x - 1 # x \in {-1,0,1}
    else:
        x = torch.randint(0,2,(1,))[0].item()
        x = 2*x - 1 # x \in {-1,1}
    return x
    
def permute_no_mid(T,N):
    perm = list(npr.permutation(T))
    perm.remove(T//2)
    return np.array(perm)[:N]

def simulate_dynamics(full_image,nframes,motion=None,init_tl=128,size=32):
    if motion is None:
        motion = simulate_motion(full_image,nframes)
    dynamic,aligned = apply_dynamics_to_full(full_image,nframes,motion,init_tl,size)
    return dynamic,aligned,motion

def simulate_motion(full_image,nframes):
    motion = []
    dyn_idx = permute_no_mid(nframes,nframes//2-1)
    for t in range(nframes):
        if t in dyn_idx:
            x = sim_posneg_one(True)
            if abs(x) > 0: y = 0
            else: y = sim_posneg_one(False)
        else: x,y = 0,0
        motion.append(torch.LongTensor([x,y]))
    motion = torch.stack(motion,dim=0).long()
    return motion

def apply_dynamics_to_full(full_image,nframes,motion,init_tl,size):
    burst,aligned = [],[]
    if isinstance(init_tl,int): x,y = init_tl,init_tl
    else: x,y = init_tl[0],init_tl[1]
    for t in range(nframes):
        mtop,mleft = list(motion[t])
        if mtop == 0 and mleft == 0: aligned.append(0)
        else: aligned.append(1)
        frame = tvF.crop(full_image,x+mtop,y+mleft,size,size)
        burst.append(frame)
    burst = torch.stack(burst,dim=0)
    aligned = torch.LongTensor(aligned)
    return burst,aligned

    
def compute_recs_psnrs(recs,clean):
    B,T = recs.shape[0],recs.shape[2]
    S = recs.shape[1] * recs.shape[2]
    recs = rearrange(recs,'b tm1 t c h w -> b (tm1 t) c h w')
    clean = repeat(clean,'b c h w -> b tile c h w',tile=S)
    psnrs = []
    for b in range(B):
        psnrs_b = torch.FloatTensor(images_to_psnrs(recs[b]+0.5,clean[b]))
        psnrs.append(psnrs_b)
    psnrs = torch.stack(psnrs,dim=0)
    psnrs = rearrange(psnrs,'b (tm1 t) -> b tm1 t',b=B,t=T)
    return psnrs

def train_logit_cog(cfg,logit,optim,data,noise_xform,n_steps=1000,init_params=None):
    results = {'acc':[]}
    P,T = 25,cfg.nframes
    use_global_motion = True
    nomotion = np.zeros((T,2)).astype(np.long)
    zeros = torch.zeros((P,T),device=cfg.device).float()
    logu = torch.log(torch.ones((P,T),device=cfg.device).float() * (1/T))
    for i in range(n_steps):

        # -- init nn params --
        nn_params = edict({'lr':1e-3,'init_params':init_params})
        if (nn_params['init_params'] is None): train_steps = cfg.cog_train_steps
        else: train_steps = cfg.cog_second_train_steps

        # -- get image data --
        full_image = data.tr[i][2].to(cfg.device)
        full_noise = noise_xform(full_image[None,:])[0]
        clean,motion,alignment,n_static,n_dynamic = [],[],[],[],[]
        init_tl_list = sample_good_init_tl(full_noise,P,32)
        global_motion = None
        for p in range(P):
            init_tl = init_tl_list[p]
            static,clean_cls,_ = simulate_dynamics(full_image,T,nomotion,init_tl,32)
            dynamic,align_cls,motion_p = simulate_dynamics(full_image,T,global_motion,init_tl,32)
            # save_image(static,f'static_{p}.png')
            # save_image(dynamic,f'dynamic_{p}.png')
            clean.append(static[0])
            motion.append(motion_p)
            alignment.append(align_cls.float().to(cfg.device))
            n_static.append(noise_xform(static))
            n_dynamic.append(noise_xform(dynamic))
            if use_global_motion: global_motion = motion_p

        # -- tensor --
        clean = torch.stack(clean,dim=0)
        motion = torch.stack(motion,dim=0)
        alignment = torch.stack(alignment,dim=0)
        n_static = torch.stack(n_static,dim=1)
        n_dynamic = torch.stack(n_dynamic,dim=1)
        # save_image(n_static,"n_static.png")
        # save_image(n_dynamic,"n_dynamic.png")

        # -- zero model grads --
        logit.zero_grad()
        optim.zero_grad()

        # -- static --
        print("static")
        cog_s = COG(UNet_small,T,cfg.device,nn_params=nn_params,train_steps=train_steps)
        cog_s.train_models_mp(n_static)
        recs = cog_s.test_models(n_static).detach().to(cfg.device) # no grad to unets
        psnrs_static = compute_recs_psnrs(recs,clean).to(cfg.device).reshape(P,-1)
        cog_s.logit = logit
        # score_static = cog_s.logit(psnrs_static/160.)
        # score_static = cog_s.logit(recs)
        score_static = cog_s.logit_consistency_score(recs,n_static)
        # save_image(recs,'rec_s.png')
        # save_image(recs[:,0,0],'rec_s_00.png')

        # -- dynamic --
        print("dynamic")
        cog_d = COG(UNet_small,T,cfg.device,nn_params=nn_params,train_steps=train_steps)
        cog_d.train_models_mp(n_dynamic)
        recs = cog_d.test_models(n_dynamic).detach().to(cfg.device) # no grad to unets
        psnrs_dynamic = compute_recs_psnrs(recs,clean).to(cfg.device).reshape(P,-1)
        cog_d.logit = logit
        # score_dynamic = cog_d.logit(psnrs_dynamic/160.)
        # score_dynamic = cog_d.logit(recs)
        score_dynamic = cog_d.logit_consistency_score(recs,n_dynamic)
        # save_image(recs,'rec_d.png')
        # save_image(recs[:,0,0],'rec_d_00.png')

        # -- compute loss --
        print(score_static)
        print(score_dynamic)
        print(alignment[[0]])
        loss = 1/T * F.binary_cross_entropy(score_static,zeros[[0]])
        loss += F.binary_cross_entropy(score_dynamic,alignment[[0]])
        # loss = F.mse_loss(score_static,logu)
        # loss += F.nll_loss(score_dynamic,torch.max(alignment,dim=1)[1])

        # -- functions of psnr --
        # print(psnrs_static[:2])
        # print(psnrs_dynamic[:2])
        psnrs_smax = torch.max(psnrs_static,dim=1)[0]
        psnrs_smin = torch.min(psnrs_static,dim=1)[0]
        extrema_static = torch.stack([psnrs_smax,psnrs_smin],dim=0)
        psnrs_dmax = torch.max(psnrs_dynamic,dim=1)[0]
        psnrs_dmin = torch.min(psnrs_dynamic,dim=1)[0]
        extrema_dynamic = torch.stack([psnrs_dmax,psnrs_dmin],dim=0)
        # print(extrema_static)
        # print(extrema_dynamic)
        print(torch.mean(extrema_static,dim=1),torch.mean(extrema_dynamic,dim=1))

        # -- sgd update --
        # loss /= P
        loss.backward()
        optim.step()

        # -- pick params --
        nn_params['init_params'] = cog_s.models[0]#pick_cog_unetp(cog_s,cog_d)

    return results,nn_params['init_params']

def pick_cog_unetp(cog_s,cog_d):
    pick = npr.permutation(2)[0]
    cog = [cog_s,cog_d][pick]
    return cog.models[0]

def test_logit_cog(cfg,logit,data,noise_xform,n_steps=1000):
    results,T = {'acc':[]},cfg.nframes
    ones = torch.ones(T,device=cfg.device)[None,:].long()
    for i in range(n_steps):

        # -- get image data --
        full_image = data.tr[i+n_steps][2].to(cfg.device)
        static,_ = simulate_dynamics(full_image,cfg.nframes,no_dynamics=True)
        dynamic,alignment = simulate_dynamics(full_image,T)
        n_static = noise_xform(static)
        n_dynamic = noise_xform(dynamic)

        # -- static --
        cog = COG(UNet_small,T,cfg.device,nn_params=None,train_steps=cfg.cog_train_steps)
        cog.train_models_mp(n_static)
        cog.logit = logit
        recs = cog.test_models(n_static).detach().to(cfg.device) # no grad to unets
        score_static = cog.logit(recs)
        # score_static = cog.logit_consistency_score(recs,n_static)

        # -- dynamic --
        cog = COG(UNet_small,T,cfg.device,nn_params=None,train_steps=cfg.cog_train_steps)
        cog.train_models_mp(n_dynamic)
        cog.logit = logit
        recs = cog.test_models(n_dynamic).detach().to(cfg.device) # no grad to unets
        score_dynamic = cog.logit(recs)
        # score_dynamic = cog.logit_consistency_score(recs,n_dynamic)

        # -- measure alignment --
        acc = F.mse_loss(ones,score_static).item()
        acc += F.mse_loss(alignment,score_dynamic).item()
        acc /= 2.
        results['acc'].append(acc)

    return results

def learn_logit_cog(cfg,data):

    # -- pick noise --
    noise_type = "g"
    cfg.noise_type = noise_type
    cfg.noise_params['g']['stddev'] = 75.
    cfg.noise_params['pn']['alpha'] = 40.0
    cfg.noise_params['pn']['readout'] = 0.0
    cfg.noise_params.ntype = cfg.noise_type
    noise_xform = get_noise_transform(cfg.noise_params,use_to_tensor=False)

    # -- get results fn --
    results_fn = COG_DIR / "learn/"
    if results_fn.exists(): results_fn.mkdir()
    results_fn = results_fn / "default.csv"

    # -- init logit model --
    T,H = cfg.nframes,cfg.nblocks
    # logit = AlignmentDetector(T).to(cfg.device)
    n_inputs,n_outputs = 9,T
    # n_inputs,n_outputs = (T-1)**2*H**2+(T-1)*H**2+H**2,T
    # n_inputs,n_outputs = (T-1)*T,T
    logit = Logistic(n_inputs,n_outputs).to(cfg.device)
    optim = torch.optim.Adam(logit.parameters(),lr=1e-2,betas=(0.9,0.99))

    # -- run learn loop --
    unetp = None
    n_epochs,n_tr_steps,n_te_steps = 50,1000,10
    results = edict({'tr_acc':[],'te_acc':[]})
    for i in range(n_epochs):
        tr_res,unetp = train_logit_cog(cfg,logit,optim,data,noise_xform,n_tr_steps,unetp)
        te_res = test_logit_cog(cfg,logit,data,noise_xform,n_te_steps)
        results.tr_acc.append(tr_res.acc)
        results.te_acc.append(te_res.acc)
    results = pd.DataFrame(results)
    results.to_csv(results_fn)
    return results


def explore_cog(cfg,data,overwrite=False):
    
    cfg.cog_train_steps = 1000
    cfg.cog_second_train_steps = 50
    cfg.nframes = 5
    # learn_logit_cog(cfg,data)
    # exit()
    # for i in range(1):
    #     iter_over_image_unet(cfg,data)
    #     # single_image_unet(cfg,data)
    #     print(f"completed run {i}.")

    cfg.nframes = 5
    cfg.nblocks = 9
    noise_str = "g25"
    record_fn = COG_DIR / f"default_{cfg.nframes}f_{cfg.nblocks}b_{noise_str}_f32_cog_2.csv"
    bss_dir = COG_DIR / f"bss"#_{cfg.nframes}f_{cfg.nblocks}b_{noise_str}_f32_cog.npy"
    # bss_dir = COG_DIR / f"bss_{cfg.nframes}f_{cfg.nblocks}b_{noise_str}_f32_cog.npy"
    # fn = "/home/gauenk/Documents/experiments/cl_gen/output/lpas/fast_unet/default_3f_3b.csv"
    overwrite = True
    if (not record_fn.exists()) or overwrite: record = run_experiment(cfg,data,record_fn,bss_dir)
    else: record = pd.read_csv(record_fn)
    # explore_fast_unet_record(cfg,record,bss_dir)
    explore_cog_record(cfg,record,bss_dir)



"""
1. discover the linear function from graph data
2. compare hyperparameters of the cog method
3. 

"""
