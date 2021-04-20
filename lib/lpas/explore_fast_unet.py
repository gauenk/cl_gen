"""
How fast can we train a unet to denoise a the same
image content with different noise patterns?
"""

# -- python imports --
import sys,os,random
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
import torchvision.utils as tv_utils
import torch.nn.functional as F
import torch.multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as tvF

# -- project code --
import settings
from pyutils.timer import Timer
from datasets import load_dataset
from datasets.transforms import get_noise_transform,get_dynamic_transform
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr,count_parameters,images_to_psnrs
from pyutils.plot import add_legend
from layers.unet import UNet_n2n,UNet_small
from .trace import activation_trace
from .scores import get_score_function
from .utils import get_block_arangements,get_ref_block_index,save_image,get_block_arangements_subset
from .cog import COG

FAST_UNET_DIR = Path(f"{settings.ROOT_PATH}/output/lpas/fast_unet/")
if not FAST_UNET_DIR.exists(): FAST_UNET_DIR.mkdir()

def train(cfg,image,clean,burst,model,optim):
    T = burst.shape[0]
    picks = list(np.r_[np.r_[:T//2],np.r_[T//2+1:T]])
    train_steps = 300
    for i in range(train_steps):

        # -- reset --
        model.zero_grad()
        optim.zero_grad()

        # -- rand in and out --
        if T == 2:
            order = npr.permutation(T)
            input_idx = order[0]
            i,j = order[1],order[1]
        else:
            input_idx = T//2
            i,j = random.sample(picks,2)
        # i,j = random.sample(list(range(burst.shape[0])),2)
        noisy = burst[[input_idx]]
        target = burst[[j]]
        
        # -- forward --
        rec = model(noisy)
        loss = F.mse_loss(rec,target)

        # -- optim step --
        loss.backward()
        optim.step()

        # save_image(rec,"fast_unet_train_rec.png",normalize=True,vrange=(-0.5,0.5))

def test(cfg,image,clean,burst,model,idx):

    T = burst.shape[0]
    # -- create results --
    results = {}

    # -- repeat along axis --
    rep = repeat(image,'c h w -> tile c h w',tile=T)

    # -- reconstruct a clean image --
    rec = model(burst)+0.5

    # -- parameters --
    params = torch.cat([param.view(-1) for param in model.parameters()])
    params_norm_mean = float(torch.norm(params).item())
    results['params_norm_mean'] = params_norm_mean

    # -- size of params for each sample's activations path --
    trace_norm = activation_trace(model,burst,'norm')
    results['trace_norm'] = trace_norm

    # -- save --
    if idx == 49 or idx == 40 or idx == 60:
        save_image(rec,f"fast_unet_rec_{idx}.png",normalize=True)

    # -- compute results --
    loss = F.mse_loss(rec,rep)
    psnr = float(np.mean(images_to_psnrs(image,rec[T//2])))
    results['mse'] = loss.item()
    results['psnr_rec'] = psnr


    psnr = float(np.mean(images_to_psnrs(rec,rep)))
    results['mse'] = loss.item()
    results['psnr_burst'] = psnr

    # -- intra and input --
    intra_input = 0
    for t in range(T):
        intra_input += F.mse_loss(rec[t],rec[T//2]).item()
        intra_input += F.mse_loss(rec[t],burst[T//2]).item()
    results['psnr_intra_input'] = intra_input

    # -- this n2n training creates a barycenter for center image  --
    bc_loss = 0
    for t in range(T):
        bc_loss += F.mse_loss(burst[t],rec[T//2]).item()
    results['psnr_bc_v1'] = bc_loss

    # -- compute psnr of clean and noisy frames --
    psnr_noisy = float(np.mean(images_to_psnrs(rep,burst+0.5)))
    results['psnr_noisy'] = psnr_noisy
    psnr_clean = float(np.mean(images_to_psnrs(rep,clean)))
    results['psnr_clean'] = psnr_clean

    # -- compute scores --
    score_fxn_names = ['lgsubset_v_ref','lgsubset','ave','lgsubset_v_indices','gaussian_ot']
    wrapped_l = []
    for name in score_fxn_names:
        score_fxn = get_score_function(name)
        wrapped_score = score_function_wrapper(score_fxn)
        if name == "gaussian_ot":
            score = wrapped_score(cfg,rec-rep).item()
        else:
            score = wrapped_score(cfg,rec).item()
        results[f"fu_{name}"] = score

    # -- on raw pixels too --
    for name in score_fxn_names:
        if name == "gaussian_ot": continue
        score_fxn = get_score_function(name)
        wrapped_score = score_function_wrapper(score_fxn)
        score = wrapped_score(cfg,burst).item()
        results[name] = score
        
    # print("Test Loss",loss.item())
    # print("Test PSNR: %2.3e" % np.mean(images_to_psnrs(rec+0.5,rep)))
    tv_utils.save_image(rec,"fast_unet_rec.png",normalize=True)
    tv_utils.save_image(burst,"fast_unet_burst.png",normalize=True)
    return results

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


def explore_cog_record(cfg,record,block_search_space_fn=None):
    REF_H = get_ref_block_index(3)
    cfg.nblocks,cfg.nframes = 5,7

    # -- load block search space --
    if block_search_space_fn:
        block_search_space = np.load(block_search_space_fn,allow_pickle=True)
    else:
        block_search_space = get_block_arangements(cfg.nblocks,cfg.nframes)

    # -- prepare data for plotting --
    P = len(record)
    psnr_clean = record['psnr_clean'].to_numpy()
    cog = record['cog'].to_numpy()

    # -- plot --
    fig,ax = plt.subplots(2,1,figsize=(8,8))
    ax[0].plot(np.arange(P),psnr_clean,'x',label='clean')
    ax[1].plot(np.arange(P),cog,'+',label='cog')
    # add_legend(ax[1],"cmp",['cog'])
    plot_dir = Path("./output/lpas/cog/")
    if not plot_dir.exists(): plot_dir.mkdir()
    plt.savefig(plot_dir / Path("cog_psnrs.png"))

    for field in record.columns:
        index = np.argmin(record[field])
        search = block_search_space[index]
        print(field,search,index)
        index = np.argmax(record[field])
        search = block_search_space[index]
        print(field,search,index)


def explore_fast_unet_record(cfg,record,block_search_space_fn=None):
    REF_H = get_ref_block_index(3)
    cfg.nblocks,cfg.nframes = 5,7

    # -- load block search space --
    if block_search_space_fn:
        block_search_space = np.load(block_search_space_fn,allow_pickle=True)
    else:
        block_search_space = get_block_arangements(cfg.nblocks,cfg.nframes)

    # -- prepare data for plotting --
    P = len(record)
    mse = record['mse'].to_numpy()
    psnrs_clean = record['psnr_clean'].to_numpy()
    psnrs_bc_v1 = record['psnr_bc_v1'].to_numpy()
    psnrs_rec = record['psnr_rec'].to_numpy()
    psnrs_burst = record['psnr_burst'].to_numpy()
    psnrs_noisy = record['psnr_noisy'].to_numpy()
    fu_ave = record['fu_ave'].to_numpy()
    ave = record['ave'].to_numpy()
    fu_lvi = record['fu_lgsubset_v_indices'].to_numpy()
    lvi = record['lgsubset_v_indices'].to_numpy()
    # pii = None
    pii = record['psnr_intra_input'].to_numpy()
    got = record['fu_gaussian_ot'].to_numpy()
    params_nm = record['params_norm_mean'].to_numpy()    
    # trace_norm = None
    trace_norm = record['params_norm_mean'].to_numpy()

    # -- rescale for plotting --
    mse -= mse.min()
    mse /= mse.max()
    mse *= psnrs_rec.max()

    pii -= pii.min()
    pii /= pii.max()
    pii *= ave.max()

    params_nm -= params_nm.min()
    params_nm /= params_nm.max()
    params_nm *= ave.max()

    # -- rescale for plotting --
    # got /= got.max()
    # got *= ave.max()

    # -- plot --
    fig,ax = plt.subplots(3,1,figsize=(8,8))
    ax[0].plot(np.arange(P),psnrs_clean,'x',label='clean')
    ax[1].plot(np.arange(P),psnrs_bc_v1,'+',label='bc')
    ax[1].plot(np.arange(P),psnrs_rec,'+',label='rec')
    # ax[1].plot(np.arange(P),mse,'+',label='mse')
    ax[1].plot(np.arange(P),psnrs_burst,'+',label='burst')
    ax[1].plot(np.arange(P),psnrs_noisy,'+',label='noisy')
    labels_2 = []
    ax[2].plot(np.arange(P),params_nm,'*',label='params_nm')
    labels_2.append('params_nm')
    if not (trace_norm is None):
        ax[2].plot(np.arange(P),trace_norm,'*',label='trace_norm')
        labels_2.append('trace_norm')
    ax[2].plot(np.arange(P),ave,'+',label='ave')
    labels_2.append('ave')
    ax[2].plot(np.arange(P),fu_ave,'+',label='fu_ave')
    labels_2.append('fu_ave')
    ax[2].plot(np.arange(P),lvi,'^',label='lvi')
    labels_2.append('lvi')
    ax[2].plot(np.arange(P),fu_lvi,'^',label='fu_lvi')
    labels_2.append('fu_lvi')
    ax[2].plot(np.arange(P),got,'x',label='got')
    labels_2.append('got')
    if not (pii is None):
        ax[2].plot(np.arange(P),pii,'x',label='pii')
        labels_2.append('pii')
    add_legend(ax[1],"cmp",['bc_v1','rec','burst','noisy'])
    add_legend(ax[2],"srch_fxn",labels_2)
    plt.savefig("./output/lpas/fast_unet/psnrs.png")

    for field in record.columns:
        index = np.argmin(record[field])
        search = block_search_space[index]
        print(field,search,index)
        index = np.argmax(record[field])
        search = block_search_space[index]
        print(field,search,index)

def run_experiment(cfg,data,record_fn,bss_fn):

    # -- setup noise --
    cfg.noise_type = 'g'
    cfg.ntype = cfg.noise_type
    cfg.noise_params.ntype = cfg.noise_type
    noise_level = 50.
    cfg.noise_params['g']['stddev'] = noise_level
    noise_level_str = f"{int(noise_level)}"
    # nconfig = get_noise_config(cfg,exp.noise_type)
    noise_xform = get_noise_transform(cfg.noise_params,use_to_tensor=False)

    # -- set configs --
    T = cfg.nframes
    H = cfg.nblocks

    # -- create our neighborhood --
    full_image = data.tr[0][2]


    clean = []
    # tl_list = [[0,0],[1,0],[0,1]]
    tl_list = np.zeros((T,2)).astype(np.int)#[[0,0],[0,0],[0,0]]
    for t in range(T):
        clean_t = []
        t,l = tl_list[t]
        for i in range(-H//2+1,H//2+1):
            for j in range(-H//2+1,H//2+1):
                clean_t.append(tvF.crop(full_image,t+128+i,l+128+j,32,32))
        clean_t = torch.stack(clean_t,dim=0)
        clean.append(clean_t)
    clean = torch.stack(clean,dim=0)
    REF_H = get_ref_block_index(cfg.nblocks)
    image = clean[T//2,REF_H]
    
    # -- normalize --
    clean -= clean.min()
    clean /= clean.max()
    save_image(clean,"fast_unet_clean.png",normalize=True)
    save_image(image,"fast_unet_image.png",normalize=True)

    # -- apply noise --
    # torch.manual_seed(123)
    noisy = noise_xform(clean)
    
    # aveT = torch.mean(burst,dim=0)
    # print("Ave MSE: %2.3e" % images_to_psnrs(aveT.unsqueeze(0),image.unsqueeze(0)))
    record = []

    gridT = torch.arange(T)
    # -- exhaustive --
    block_search_space = get_block_arangements_subset(cfg.nblocks,cfg.nframes,tcount=3)
    # block_search_space = get_block_arangements(cfg.nblocks,cfg.nframes)
    # -- random subset --
    use_rand = True
    if use_rand:
        if bss_fn.exists() and False:
            print(f"Reading bss {bss_fn}")
            block_search_space = np.load(bss_fn,allow_pickle=True)
        else:
            bss = block_search_space
            print(f"Original block search space: [{len(bss)}]")
            if len(block_search_space) >= 100:
                rand_blocks = random.sample(list(block_search_space),100)
                block_search_space = [np.array([REF_H]*T),] # include gt
                block_search_space.extend(rand_blocks)
            bss = block_search_space
            print(f"Writing block search space: [{bss_fn}]")
            np.save(bss_fn,np.array(block_search_space))
    print(f"Search Space Size: {len(block_search_space)}")
    
    idx = 0
    clean = clean.to(cfg.device)
    noisy = noisy.to(cfg.device)
    for prop in tqdm(block_search_space):
        # -- fetch --
        clean_prop = clean[gridT,prop]
        noisy_prop = noisy[gridT,prop]
        if idx == 49 or idx == 40 or idx == 40:
            save_image(noisy_prop,f"noisy_prop_{idx}.png",normalize=True)
            save_image(clean_prop,f"clean_prop_{idx}.png",normalize=True)

        # -- compute again --
        train_steps = 500
        cog = COG(UNet_small,T,noisy.device,nn_params=None,train_steps=train_steps)
        cog.train_models(noisy_prop)
        recs = cog.test_models(noisy_prop)
        score = cog.operator_consistency(recs,noisy_prop)
        results = fill_results(cfg,image,clean_prop,noisy_prop,None,idx)
        results['cog'] = score
        print(score,prop)

        # -- compute --
        # model = [UNet_small(3),UNet_small(3)]
        # model = UNet_small(3) # UNet_n2n(1)
        # cfg.init_lr = 1e-4
        # optim = torch.optim.Adam(model.parameters(),lr=cfg.init_lr,betas=(0.9,0.99))
        # train(cfg,image,clean_prop,noisy_prop,model,optim)
        # results = test(cfg,image,clean_prop,noisy_prop,model,idx)


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
    noise_level = 50./255.

    # -- poisson noise --
    noise_type = "pn"
    cfg.noise_type = noise_type
    cfg.noise_params['pn']['alpha'] = 4.0
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
    acc,nruns = 0,3
    train_steps = 100
    for i in range(nruns):

        # -- create COG --
        print("aligned")
        cog = COG(UNet_small,T,image.device,nn_params=None,train_steps=train_steps)
        cog.train_models_mp(noisy)
        recs = cog.test_models(noisy)
        score_align = cog.operator_consistency(recs,noisy)
        save_image(noisy,"noisy.png")
        save_image(recs,"recs_aligned.png")
        # noisy_rep = repeat(noisy,'t c h w -> tile t c h w',tile=T)
        # score_align = cog.compute_consistency(recs,noisy_rep)
        
        print("misaligned")
        cog = COG(UNet_small,T,image.device,nn_params=None,train_steps=train_steps)
        cog.train_models_mp(m_noisy)
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
    acc = acc.item() / float(nruns)
    if not (queue is None): queue.put((lacc,acc))
    # print(F.mse_loss(m_clean[0][:,32//2,32//2],clean[0][:,32//2,32//2]).item())
    return acc


    # -- model all --
    print("-- All Aligned --")
    model = UNet_small(3) # UNet_n2n(1)
    cfg.init_lr = 1e-4
    optim = torch.optim.Adam(model.parameters(),lr=cfg.init_lr,betas=(0.9,0.99))
    train(cfg,image,clean,noisy,model,optim)
    results = test(cfg,image,clean,noisy,model,0)
    rec = model(noisy)+0.5
    save_image(rec,"rec_all.png",normalize=True)
    print("Single Image Unet:")
    print(images_to_psnrs(clean,rec))
    print(images_to_psnrs(rec-0.5,noisy))
    print(images_to_psnrs(rec[[0]],rec[[1]]))
    print(images_to_psnrs(rec[[0]],rec[[2]]))
    print(images_to_psnrs(rec[[1]],rec[[2]]))

    og_clean,og_noisy = clean.clone(),noisy.clone()
    clean[0] = image_mis
    noisy[0] = torch.normal(clean[0]-0.5,noise_level)
    # -- model all --
    print("-- All Misligned --")
    model = UNet_small(3) # UNet_n2n(1)
    cfg.init_lr = 1e-4
    optim = torch.optim.Adam(model.parameters(),lr=cfg.init_lr,betas=(0.9,0.99))
    train(cfg,image,clean,noisy,model,optim)
    results = test(cfg,image,clean,noisy,model,0)
    rec = model(noisy)+0.5
    save_image(rec,"rec_all.png",normalize=True)
    print("Single Image Unet:")
    print(images_to_psnrs(clean,rec))
    print(images_to_psnrs(rec-0.5,noisy))
    print(images_to_psnrs(rec[[0]],rec[[1]]))
    print(images_to_psnrs(rec[[0]],rec[[2]]))
    print(images_to_psnrs(rec[[1]],rec[[2]]))


    for j in range(3):
        # -- data --
        noisy1 = torch.stack([noisy[0],noisy[1]],dim=0)
        clean1 = torch.stack([clean[0],clean[1]],dim=0)
        
        noisy2 = torch.stack([noisy[1],noisy[2]],dim=0)
        clean2 = torch.stack([clean[1],clean[2]],dim=0)

        noisy3 = torch.stack([og_noisy[0],noisy[1]],dim=0)
        clean3 = torch.stack([og_clean[0],clean[1]],dim=0)

    
        # -- model 1 --
        model = UNet_small(3) # UNet_n2n(1)
        cfg.init_lr = 1e-4
        optim = torch.optim.Adam(model.parameters(),lr=cfg.init_lr,betas=(0.9,0.99))
        train(cfg,image,clean1,noisy1,model,optim)
        results = test(cfg,image,clean1,noisy1,model,0)
        rec = model(noisy1)+0.5
        xrec = model(noisy2)+0.5
        save_image(rec,"rec1.png",normalize=True)
        print("[misaligned] Single Image Unet:",
              images_to_psnrs(clean1,rec),
              images_to_psnrs(clean2,xrec),
              images_to_psnrs(rec-0.5,noisy1),
              images_to_psnrs(xrec-0.5,noisy2),
              images_to_psnrs(rec[[0]],rec[[1]]),
              images_to_psnrs(xrec[[0]],xrec[[1]]),
              images_to_psnrs(xrec[[0]],rec[[1]]),
              images_to_psnrs(xrec[[1]],rec[[0]])
        )

        # -- model 2 --
        model = UNet_small(3) # UNet_n2n(1)
        cfg.init_lr = 1e-4
        optim = torch.optim.Adam(model.parameters(),lr=cfg.init_lr,betas=(0.9,0.99))
        train(cfg,image,clean2,noisy2,model,optim)
        results = test(cfg,image,clean2,noisy2,model,0)
        rec = model(noisy2)+0.5
        xrec = model(noisy1)+0.5
        save_image(rec,"rec2.png",normalize=True)
        print("[aligned] Single Image Unet:",
              images_to_psnrs(clean2,rec),
              images_to_psnrs(clean1,xrec),
              images_to_psnrs(rec-0.5,noisy2),
              images_to_psnrs(xrec-0.5,noisy1),
              images_to_psnrs(rec[[0]],rec[[1]]),
              images_to_psnrs(xrec[[0]],xrec[[1]]),
              images_to_psnrs(xrec[[0]],rec[[1]]),
              images_to_psnrs(xrec[[1]],rec[[0]])
        )

        # -- model 3 --
        model = UNet_small(3) # UNet_n2n(1)
        cfg.init_lr = 1e-4
        optim = torch.optim.Adam(model.parameters(),lr=cfg.init_lr,betas=(0.9,0.99))
        train(cfg,image,clean3,noisy3,model,optim)
        results = test(cfg,image,clean3,noisy3,model,0)
        rec = model(noisy3)+0.5
        rec_2 = model(noisy2)+0.5
        rec_1 = model(noisy1)+0.5
        save_image(rec,"rec1.png",normalize=True)
        print("[aligned (v3)] Single Image Unet:")
        print("clean-rec",images_to_psnrs(clean3,rec))
        print("clean1-rec1",images_to_psnrs(clean1,rec_1))
        print("clean2-rec2",images_to_psnrs(clean2,rec_2))
        print("rec-noisy3",images_to_psnrs(rec-0.5,noisy3))
        print("rec1-noisy1",images_to_psnrs(rec_1-0.5,noisy1))
        print("rec2-noisy2",images_to_psnrs(rec_2-0.5,noisy2))
        print("[v3]: rec0-rec1",images_to_psnrs(rec[[0]],rec[[1]]))
        print("[v1]: rec0-rec1",images_to_psnrs(rec_1[[0]],rec_1[[1]]))
        print("[v2]: rec0-rec1",images_to_psnrs(rec_2[[0]],rec_2[[1]]))
        print("[v1-v2](a):",images_to_psnrs(rec_1[[1]],rec_2[[1]]))
        print("[v1-v2](b):",images_to_psnrs(rec_1[[1]],rec_2[[0]]))
        print("[v1-v2](c):",images_to_psnrs(rec_1[[0]],rec_2[[1]]))
        print("[v1-v2](d):",images_to_psnrs(rec_1[[0]],rec_2[[0]]))
        print("-"*20)
        print("[v2-v3](a):",images_to_psnrs(rec_2[[1]],rec[[1]]))
        print("[v2-v3](b):",images_to_psnrs(rec_2[[1]],rec[[0]]))
        print("[v2-v3](c):",images_to_psnrs(rec_2[[0]],rec[[1]]))
        print("[v2-v3](d):",images_to_psnrs(rec_2[[0]],rec[[0]]))
        print("-"*20)
        print("[v1-v3](a):",images_to_psnrs(rec_1[[1]],rec[[1]]))
        print("[v1-v3](b):",images_to_psnrs(rec_1[[1]],rec[[0]]))
        print("[v1-v3](c):",images_to_psnrs(rec_1[[0]],rec[[1]]))
        print("[v1-v3](d):",images_to_psnrs(rec_1[[0]],rec[[0]]))
        print("-"*20)
        print("rec0-rec1",images_to_psnrs(rec[[0]],rec[[1]]))


def explore_cog(cfg,data):
    for i in range(1):
        iter_over_image_unet(cfg,data)
        # single_image_unet(cfg,data)
        print(f"completed run {i}.")

def fast_unet(cfg,data,overwrite=False):
    

    cfg.nframes = 5
    explore_cog(cfg,data)
    exit()

    cfg.nframes = 7
    cfg.nblocks = 5
    noise_str = "g50"
    record_fn = FAST_UNET_DIR / f"default_{cfg.nframes}_{cfg.nblocks}b_{noise_str}_f32_cog_trPair.csv"
    bss_fn = FAST_UNET_DIR / f"bss_{cfg.nframes}f_{cfg.nblocks}b_{noise_str}_f32_cog_trPair.npy"
    # fn = "/home/gauenk/Documents/experiments/cl_gen/output/lpas/fast_unet/default_3f_3b.csv"
    if (not record_fn.exists()) or overwrite: record = run_experiment(cfg,data,record_fn,bss_fn)
    else: record = pd.read_csv(record_fn)
    # explore_fast_unet_record(cfg,record,bss_fn)
    explore_cog_record(cfg,record,bss_fn)

    


