
# -- python imports --
import time,os
import numpy as np
import pandas as pd
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- pytorch imports --
import torch

# -- project imports --
import settings
import cache_io
from pyutils import tile_patches,save_image,torch_to_numpy,edict_torch_to_numpy
from pyutils.vst import anscombe
from patch_search import get_score_function
from datasets.wrap_image_data import load_image_dataset,sample_to_cuda

# -- cuda profiler --
import nvtx

# -- [align] package imports --
import align.nnf as nnf
from align.combo.eval_scores import EvalBlockScores
from align.combo.optim import AlignOptimizer
from align.xforms import align_from_pix,flow_to_pix,create_isize,pix_to_flow,align_from_flow,flow_to_blocks

# -- [local] package imports --
from .exp_utils import *
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def remove_center_frame(frames):
    nframes = frames.shape[0]
    nc_frames = torch.cat([frames[:nframes//2],frames[nframes//2+1:]],dim=0)
    return nc_frames

def check_parameters(nblocks,patchsize):
    even_blocks = nblocks % 2 == 0
    even_ps = patchsize % 2 == 0
    if even_blocks or even_ps:
        print("Even number of blocks or patchsizes. We recommend odd so a center exists.")

@nvtx.annotate("test_nnf", color="purple")
def execute_experiment(cfg):

    # -- init exp! --
    print("RUNNING EXP.")
    print(cfg)

    # -- create results record to save --
    dims={'batch_results':None,
          'batch_to_record':None,
          'record_results':{'default':0},
          'stack':{'default':0},
          'cat':{'default':0}}
    record = cache_io.ExpRecord(dims)

    # -- set random seed --
    set_seed(cfg.random_seed)	

    # -- load dataset --
    data,loaders = load_image_dataset(cfg)
    image_iter = iter(loaders.tr)    
    
    # -- get score function --
    score_fxn_ave = get_score_function("ave")
    score_fxn_bs = get_score_function(cfg.score_fxn_name)

    # -- some constants --
    NUM_BATCHES = 3
    nframes,nblocks = cfg.nframes,cfg.nblocks 
    patchsize = cfg.patchsize
    ppf = cfg.dynamic_info.ppf
    check_parameters(nblocks,patchsize)

    # -- create evaluator for ave; simple --
    iterations,K = 1,1
    subsizes = []
    block_batchsize = 256
    eval_ave_simp = EvalBlockScores(score_fxn_ave,"ave",patchsize,block_batchsize,None)

    # -- create evaluator for ave --
    iterations,K = 1,1
    subsizes = []
    eval_ave = EvalBlockScores(score_fxn_ave,"ave",patchsize,block_batchsize,None)

    # -- create evaluator for bootstrapping --
    block_batchsize = 64
    eval_prop = EvalBlockScores(score_fxn_bs,"bs",patchsize,block_batchsize,None)

    # -- iterate over images --
    for image_bindex in range(NUM_BATCHES):

        print("-="*30+"-")
        print(f"Running image batch index: {image_bindex}")
        print("-="*30+"-")
        torch.cuda.empty_cache()

        # -- sample & unpack batch --
        sample = next(image_iter)
        sample_to_cuda(sample)

        dyn_noisy = sample['noisy'] # dynamics and noise
        dyn_clean = sample['burst'] # dynamics and no noise
        static_noisy = sample['snoisy'] # no dynamics and noise
        static_clean = sample['sburst'] # no dynamics and no noise
        flow_gt = sample['flow']
        image_index = sample['index']
        if cfg.noise_params.ntype == "pn":
            dyn_noisy = anscombe.forward(dyn_noisy)

        # -- shape info --
        T,B,C,H,W = dyn_noisy.shape
        isize = edict({'h':H,'w':W})
        ref_t = nframes//2
        npix = H*W

        # -- create results dict --
        runtimes = edict()

        # -- groundtruth flow --
        flow_gt_rs = rearrange(flow_gt,'i tm1 two -> i 1 tm1 two')
        blocks_gt = flow_to_blocks(flow_gt_rs,nblocks)
        flow_gt = repeat(flow_gt,'i tm1 two -> i p tm1 two',p=npix)
        aligned_of = align_from_flow(dyn_clean,flow_gt,nblocks,isize=isize)
        pix_gt = flow_to_pix(flow_gt.clone(),isize=isize)
        runtimes.of = 0. # given

        # -- compute nearest neighbor fields --
        start_time = time.perf_counter()
        shape_str = 't b h w two -> b (h w) t two'
        nnf_vals,nnf_pix = nnf.compute_burst_nnf(dyn_clean,ref_t,patchsize)
        nnf_pix_best = torch.LongTensor(rearrange(nnf_pix[...,0,:],shape_str))
        nnf_pix_best = torch.LongTensor(nnf_pix_best)
        pix_nnf = nnf_pix_best.clone()
        flow_nnf = pix_to_flow(nnf_pix_best)
        aligned_nnf = align_from_pix(dyn_clean,nnf_pix_best,nblocks)
        runtimes.nnf = time.perf_counter() - start_time

        # -- compute proposed search of nnf --
        start_time = time.perf_counter()
        split_vals,split_pix = nnf.compute_burst_nnf(dyn_noisy,ref_t,patchsize)
        # split_pix = np.copy(nnf_pix)
        split_pix_best = torch.LongTensor(rearrange(split_pix[...,0,:],shape_str))
        split_pix_best = torch.LongTensor(split_pix_best)
        pix_split = split_pix_best.clone()
        flow_split = pix_to_flow(split_pix_best)
        aligned_split = align_from_pix(dyn_clean,split_pix_best,nblocks)
        runtimes.split = time.perf_counter() - start_time

        # -- compute complex ave --
        iterations,K = 0,1
        subsizes = []
        print("[complex] Ave loss function")
        start_time = time.perf_counter()
        optim = AlignOptimizer("v3")
        flow_ave = optim.run(dyn_noisy,patchsize,eval_ave,
                             nblocks,iterations,subsizes,K)
        # flow_ave = flow_gt.clone()
        pix_ave = flow_to_pix(flow_ave.clone(),isize=isize)
        aligned_ave = align_from_flow(dyn_clean,flow_ave,nblocks,isize=isize)
        runtimes.ave = time.perf_counter() - start_time

        # -- compute simple ave --
        iterations,K = 0,1
        subsizes = []
        print("[simple] Ave loss function")
        start_time = time.perf_counter()
        optim = AlignOptimizer("v3")
        # flow_ave_simp = optim.run(dyn_noisy,patchsize,eval_ave_simp,
        #                      nblocks,iterations,subsizes,K)
        flow_ave_simp = flow_ave.clone().cpu()
        pix_ave_simp = flow_to_pix(flow_ave_simp.clone(),isize=isize)
        aligned_ave_simp = align_from_flow(dyn_clean,flow_ave_simp,nblocks,isize=isize)
        runtimes.ave_simp = time.perf_counter() - start_time

        # -- compute proposed search of nnf --
        # iterations,K = 50,3
        # subsizes = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
        #iterations,K = 1,nblocks**2
        # K is a function of noise level.
        # iterations,K = 1,nblocks**2
        iterations = 1
        if cfg.nframes == 3:
            K = nblocks**2
            subsizes = [cfg.nframes]
        elif cfg.nframes == 5:
            K = 2*nblocks
            subsizes = [cfg.nframes]
        elif cfg.nframes <= 20:
            K = 2*nblocks
            subsizes = [2,]*cfg.nframes
        else:
            K = nblocks
            subsizes = [2,]*cfg.nframes
        # iterations,K = 1,2*nblocks#**2
        # subsizes = [3]#,3,3,3,3,3,3,3,3,3]
        # subsizes = [3,3,3,3,3,3,3,]
        # subsizes = [3,3,3,3,3,3,3,3]
        # subsizes = [nframes]
        print("[Bootstrap] loss function")
        start_time = time.perf_counter()
        optim = AlignOptimizer("v3")
        flow_est = optim.run(dyn_noisy,patchsize,eval_prop,
                             nblocks,iterations,subsizes,K)
        # flow_est = flow_gt.clone()
        pix_est = flow_to_pix(flow_est.clone(),isize=isize)
        aligned_est = align_from_flow(dyn_clean,flow_est,patchsize,isize=isize)
        runtimes.est = time.perf_counter() - start_time
        # aligned_est = aligned_of.clone()
        # time_est = 0.

        # -- format results --
        #pad = 2*(nframes-1)*ppf+4
        pad = 3#2*(nframes-1)*ppf+4
        isize = edict({'h':H-pad,'w':W-pad})
        frames = edict({
            'of':aligned_of,'nnf':aligned_nnf,'split':aligned_split,
            'ave_simp':aligned_ave_simp,'ave':aligned_ave,
            'est':aligned_est,'clean':static_clean,
        })
        flows = edict({
            'gt':flow_gt,'nnf':flow_nnf,'split':flow_split,
            'ave':flow_ave,'ave_simp':flow_ave_simp,'est':flow_est
        })
        pixs = edict({
            'gt':pix_gt,'nnf':pix_nnf,'split':pix_split,
            'ave':pix_ave,'ave_simp':pix_ave_simp,'est':pix_est
        })

        # -- flows to numpy --
        is_even = cfg.frame_size%2 == 0
        mid_pix = cfg.frame_size*cfg.frame_size//2 + (cfg.frame_size//2)*is_even
        mid_pix = 32*10+23
        flows_np = edict_torch_to_numpy(flows)
        pixs_np = edict_torch_to_numpy(pixs)

        # -- End-Point-Errors --
        epes_of = compute_flows_epe(flows)
        flows.gt = flows.nnf
        epes_nnf = compute_flows_epe(flows)
        nnf_acc = compute_nnf_acc(flows)

        # -- PSNRs --
        frames = remove_frame_centers(frames)
        psnrs = compute_frames_psnr(frames,isize)

        # -- print report ---
        print("\n"*3) # banner
        print("-"*25 + " Results " + "-"*25)
        print_dict_ndarray_0_midpix(flows_np,mid_pix)
        print_dict_ndarray_0_midpix(pixs_np,mid_pix)
        print_runtimes(runtimes)
        print_verbose_psnrs(psnrs)
        print_delta_summary_psnrs(psnrs)
        print_verbose_epes(epes_of,epes_nnf)
        print_nnf_acc(nnf_acc)
        print_summary_epes(epes_of,epes_nnf)
        print_summary_psnrs(psnrs)

        # -- append to record --
        psnrs = edict_torch_to_numpy(psnrs)
        epes_of = edict_torch_to_numpy(epes_of)
        epes_nnf = edict_torch_to_numpy(epes_nnf)
        nnf_acc = edict_torch_to_numpy(nnf_acc)
        # image_index = np.array([image_index])
        image_index  = torch_to_numpy(image_index)

        batch_results = {'runtimes':runtimes,
                         'psnrs':psnrs,
                         'epes_of':epes_of,
                         'epes_nnf':epes_nnf,                         
                         'nnf_acc':nnf_acc}
                         
        # record.append(batch_results)
        # print(record.record)
        # print("-"*20)
        batch_results = flatten_internal_dict(batch_results)
        format_fields(batch_results,image_index)
        print("inner.")
        for key,value in batch_results.items():
            print(key,value.shape)
        # tmp_df = pd.DataFrame(batch_results)
        # print("-"*10)
        # print(tmp_df)
        # print(batch_results)
        # print(len(batch_results['psnrs']))
        # print("-"*10)
        record.append(batch_results)
    # print("\n"*3)
    # print("-"*20)
    # print(record.record)
    # print("-"*20)    
    # print("\n"*3)
    # record.stack_record()
    record.cat_record()
    # print("\n"*3)
    # print("-"*20)
    # print(record.record)
    # print("-"*20)    
    print("\n"*3)

    print("\n"*3)
    print("-"*20)
    # df = pd.DataFrame().append(record.record,ignore_index=True)
    for key,val in record.record.items():
        print(key,val.shape)
    # print(df)
    print("-"*20)    
    print("\n"*3)

    return record.record
    


def get_result_method_names(results):
    return list(results['psnrs'].keys())

def flatten_internal_dict(results):
    flattened = []
    methods = get_result_method_names(results)
    # print("methods.")
    # print(methods)
    method_dict = {key:{} for key in methods}
    # print("\n"*3)
    # print("results.")
    # print("\n"*3)
    for key_a,value_a in results.items():
        for key_b,value_b in value_a.items():
            # print(key_a,key_b,value_b)
            # if isinstance(value_b,list):
            #     if torch.is_tensor(value_b[0]):
            #         value_b  = torch.stack(value_b,dim=0)
            #     else:
            #         value_b = torch.Tensor(value_b)
            #     print(value_b.shape)
            # if isinstance(value_b,np.ndarray):
            #     value_b = np.array(value_b)
            #     print(value_b.shape)
            method_dict[key_b][key_a] = value_b
            method_dict[key_b]['methods'] = key_b
    # print("\n"*3)
    # print("method_dict")
    # print("\n"*3)
    for key_a,value_a in method_dict.items():
        # print("-"*30)
        flattened.append(value_a)
        # for key_b,value_b in value_a.items():
        #     print(key_a,key_b,value_b)
    flattened = pd.DataFrame(flattened)
    # print(flattened)
    # print(flattened.columns)
    # for key in list(flattened.columns):
    #     print(flattened[key])
    dicts = flattened.to_dict('records')
    # print("-"*30)
    mgrouped = {}
    for elem in dicts:
        for key,value in elem.items():
            if key in mgrouped:
                mgrouped[key].append(value)
            else:
                mgrouped[key] = [value]
    # print(mgrouped)
    # print("-"*30)
    return mgrouped


def format_fields(mgrouped,index):

    # -- list keys --
    print(list(mgrouped.keys()))

    # -- get reference shapes --
    psnrs = mgrouped['psnrs']
    psnrs = np.stack(psnrs,axis=0)
    nmethods,nframes,batchsize = psnrs.shape

    # -- psnrs --
    print("psnrs.shape: ",psnrs.shape)
    psnrs = rearrange(psnrs,'m t i -> (m i) t')
    print("psnrs.shape: ",psnrs.shape)
    mgrouped['psnrs'] = psnrs

    # -- methods --
    methods = np.array(mgrouped['methods'])
    methods = repeat(methods,'m -> (m i)',i=batchsize)
    # rmethods = np.repeat(methods,batchsize).reshape(nmethods,batchsize)
    # tmethods = np.tile(rmethods,nframes).reshape(nmethods,nframes,batchsize)
    # methods = tmethods.ravel()
    print("methods.shape: ",methods.shape)
    mgrouped['methods'] = methods

    # -- runtimes --
    runtimes = np.array(mgrouped['runtimes'])
    runtimes = repeat(runtimes,'m -> (m i)',i=batchsize)
    # rruntimes = np.repeat(runtimes,batchsize).reshape(nmethods,batchsize)
    # truntimes = np.tile(rruntimes,nframes).reshape(nmethods,nframes,batchsize)
    # runtimes = truntimes.ravel()
    print("runtimes.shape: ",runtimes.shape)
    mgrouped['runtimes'] = runtimes

    # -- epes_of --
    epes_of = np.array(mgrouped['epes_of'])
    epes_of = rearrange(epes_of,'m t i -> (m i) t')
    print("epes_of.shape: ",epes_of.shape)
    mgrouped['epes_of'] = epes_of

    # -- epes_nnf --
    epes_nnf = np.array(mgrouped['epes_nnf'])
    epes_nnf = rearrange(epes_nnf,'m t i -> (m i) t')
    print("epes_nnf.shape: ",epes_nnf.shape)
    mgrouped['epes_nnf'] = epes_nnf

    # -- nnf_acc --
    nnf_acc = np.array(mgrouped['nnf_acc'])
    nnf_acc = rearrange(nnf_acc,'m t i -> (m i) t')
    print("nnf_acc.shape: ",nnf_acc.shape)
    mgrouped['nnf_acc'] = nnf_acc

    # -- index --
    index = repeat(index,'i 1 -> (m i)',m=nmethods)
    print("index.shape: ",index.shape)    
    mgrouped['image_index'] = index


    # -- test --
    df = pd.DataFrame().append(mgrouped,ignore_index=True)
    return df
