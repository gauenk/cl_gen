
# -- python imports --
import sys
import numpy as np
from einops import repeat,rearrange

# -- pytorch imports --
import torch

# -- project imports --
from easydict import EasyDict as edict
from datasets.wrap_image_data import dict_to_device
from pyutils import print_tensor_stats

from .log_learn import get_train_log_info,print_train_log_info,get_test_log_info

def train_model(cfg,model,loss_fxn,optim,data_loader,sim_fxn):

    tr_info = []
    # nbatches = len(data_loader)
    # nbatches = 500
    nbatches = 500
    data_iter = iter(data_loader)
    # for batch_iter,sample in enumerate(data_loader):
    for batch_iter in range(nbatches):

        # -- sample from iterator --
        sample = next(data_iter)

        # -- unpack sample --
        device = f'cuda:{cfg.gpuid}'
        dict_to_device(sample,device)
        dyn_noisy = sample['noisy'] # dynamics and noise
        noisy = dyn_noisy # alias
        dyn_clean = sample['burst']-0.5 # dynamics and no noise
        static_noisy = sample['snoisy'] # no dynamics and noise
        static_clean = sample['sburst']-0.5 # no dynamics and no noise
        flow_gt = sample['flow']
        image_index = sample['index']
        if cfg.noise_params.ntype == "pn" or cfg.use_anscombe:
            dyn_noisy = anscombe.forward(dyn_noisy)

        # -- shape info --
        T,B,C,H,W = dyn_noisy.shape
        nframes = T
        isize = edict({'h':H,'w':W})
        ref_t = nframes//2
        npix = H*W

        # -- global to pix --
        npix = H*W
        flow_gt = repeat(flow_gt,'b tm1 two -> b p tm1 two',p=npix)

        # -- create sim images --
        gt_info = {'dyn_clean':dyn_clean,
                   'flow':flow_gt,'static_noisy':static_noisy}
        sims,masks,aligned,flow = sim_fxn(dyn_noisy,cfg.sim_type,gt_info=gt_info)

        # -- create inputs and outputs --
        inputs = torch.cat([noisy[:T//2],sims[[0]],noisy[T//2+1:]],dim=0)
        target = sims[1]

        # -- reset gradient --
        model.zero_grad()
        optim.zero_grad()

        # -- forward pass --
        output = model(inputs,cfg.noise_params) # 
        if isinstance(output,tuple): denoised,denoised_frames = output
        else: denoised,denoised_frames = output,None

        # -- compute loss --
        loss = loss_fxn(denoised,target,denoised_frames,cfg.global_step)

        # print("-"*20)
        # print_tensor_stats("denoised",denoised)
        # print_tensor_stats("sim",sims)
        # print_tensor_stats("dyn_noisy",dyn_noisy)
        # print_tensor_stats("aligned",aligned)
        # print_tensor_stats("dyn_clean",dyn_clean)

        # -- backward --
        loss.backward()
        optim.step()

        # -- log --
        if batch_iter % cfg.train_log_interval == 0:
            info = get_train_log_info(cfg,model,denoised,loss,dyn_noisy,
                                      dyn_clean,sims,masks,aligned,
                                      flow,flow_gt)
            info['global_iter'] = cfg.global_step
            info['batch_iter'] = batch_iter
            info['mode'] = 'train'
            info['loss'] = loss.item()
            print_train_log_info(info,nbatches)

            tr_info.append(info)
            
        # -- update global step --
        cfg.global_step += 1

        # -- print update --
        sys.stdout.flush()
            
    return tr_info

def test_model(cfg,model,test_loader,loss_fxn,epoch):

    model = model.to(cfg.device)
    test_iter = iter(test_loader)
    nbatches,D = 25,len(test_iter) 
    # nbatches = 2
    # nbatches = D
    nbatches = nbatches if D > nbatches else D
    psnrs = np.zeros( ( nbatches, cfg.batch_size ) )
    use_record = False
    te_info = []

    with torch.no_grad():
        for batch_iter in range(nbatches):

            # -- load data --
            device = f'cuda:{cfg.gpuid}'
            sample = next(test_iter)
            dict_to_device(sample,device)
            
            # -- unpack --
            dyn_noisy = sample['noisy']
            dyn_clean = sample['burst'] - 0.5
            flow_gt = sample['flow']
            nframes = dyn_clean.shape[0]
            clean = dyn_clean[nframes//2]
            T,B,C,H,W = dyn_noisy.shape

            # -- denoise image --
            output = model(dyn_noisy,cfg.noise_params) # 
            if isinstance(output,tuple): denoised,denoised_frames = output
            else: denoised,denoised_frames = output,None

            # -- compute gt loss --
            loss = loss_fxn(denoised,clean,denoised_frames,cfg.global_step)

            # -- log info --
            info = get_test_log_info(cfg,model,denoised,loss,dyn_noisy,dyn_clean)
            info['global_iter'] = cfg.global_step
            info['batch_iter'] = batch_iter
            info['mode'] = 'test'
            info['loss'] = loss.item()
            te_info.append(info)
            
            # -- print to screen --
            if batch_iter % cfg.test_log_interval == 0:
                psnr = info['image_psnrs'].mean().item()
                print("[%d/%d] Test PSNR: %2.2f" % (batch_iter,nbatches,psnr))
                
            # -- print update --
            sys.stdout.flush()


    return te_info

