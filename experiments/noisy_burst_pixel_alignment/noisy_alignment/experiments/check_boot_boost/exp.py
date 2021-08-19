
# -- python imports --
import time,os,copy
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
from align.combo import EvalBlockScores,EvalBootBlockScores
from align.combo.optim import AlignOptimizer
from align.xforms import align_from_pix,flow_to_pix,create_isize,pix_to_flow,align_from_flow,flow_to_blocks
from align.combo.optim.v3 import get_boot_hyperparams
from align.combo.optim.v3._utils import get_ref_block,exh_block_range,init_optim_block
from align.combo.optim.v3._blocks import get_search_blocks

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
    score_fxn_bsl = get_score_function("bootstrapping_limitB")

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
    eval_plimb = EvalBootBlockScores(score_fxn_bsl,score_fxn_bs,"bsl",
                                     patchsize,block_batchsize,None)
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
        tl_index = sample['tl_index']
        rng_state = sample['rng_state']
        if cfg.noise_params.ntype == "pn":
            dyn_noisy = anscombe.forward(dyn_noisy)

        # -- shape info --
        T,B,C,H,W = dyn_noisy.shape
        isize = edict({'h':H,'w':W})
        ref_t = nframes//2
        nimages,npix,nframes = B,H*W,T

        # -- create results dict --
        flows = edict()
        aligned = edict()
        runtimes = edict()
        optimal_scores = edict() # score function at optimal

        # -- groundtruth flow --
        flow_gt_rs = rearrange(flow_gt,'i tm1 two -> i 1 tm1 two')
        blocks_gt = flow_to_blocks(flow_gt_rs,nblocks)
        flows.of = repeat(flow_gt,'i tm1 two -> i p tm1 two',p=npix)
        aligned.of = align_from_flow(dyn_clean,flows.of,nblocks,isize=isize)
        runtimes.of = 0. # given
        optimal_scores.of = np.zeros((nimages,npix,nframes)) # clean target is zero
        aligned.clean = static_clean

        # -- compute nearest neighbor fields [global] --
        start_time = time.perf_counter()
        shape_str = 't b h w two -> b (h w) t two'
        nnf_vals,nnf_pix = nnf.compute_burst_nnf(dyn_clean,ref_t,patchsize)
        nnf_pix_best = torch.LongTensor(rearrange(nnf_pix[...,0,:],shape_str))
        nnf_pix_best = torch.LongTensor(nnf_pix_best)
        flows.nnf = pix_to_flow(nnf_pix_best)
        aligned.nnf = align_from_pix(dyn_clean,nnf_pix_best,nblocks)
        runtimes.nnf = time.perf_counter() - start_time
        optimal_scores.nnf = np.zeros((nimages,npix,nframes)) # clean target is zero

        # -- compute proposed search of nnf --
        print("[Bootstrap] loss function")
        iterations = 1
        K,subsizes = get_boot_hyperparams(cfg.nframes,cfg.nblocks)
        start_time = time.perf_counter()
        optim = AlignOptimizer("v3")
        flows.est = optim.run(dyn_noisy,patchsize,eval_prop,
                             nblocks,iterations,subsizes,K)
        # flows.est = flows.of.clone()
        aligned.est = align_from_flow(dyn_clean,flows.est,patchsize,isize=isize)
        runtimes.est = time.perf_counter() - start_time
        
        # -- load adjacent blocks --
        nframes,nimages,ncolor,h,w = dyn_noisy.shape
        nsegs = 64#h*w
        brange = exh_block_range(nimages,nsegs,nframes,nblocks)
        ref_block = get_ref_block(nblocks)
        curr_blocks = init_optim_block(nimages,nsegs,nframes,nblocks)
        curr_blocks = curr_blocks[:,:,None,:]
        frames = repeat(np.arange(2),'t -> i s t',i=nimages,s=nsegs)
        search_blocks = get_search_blocks(frames,brange,curr_blocks,f'cuda:{cfg.gpuid}')
        print("search_blocks.shape ",search_blocks.shape)
        
        # -- compute bootstrapping for the batch --
        est = edict()
        plimb = edict()

        print("curr_blocks.shape ",curr_blocks.shape)
        eval_prop.score_fxn_name = ""
        eval_prop.score_cfg.bs_type = "full"
        scores,scores_t,blocks = eval_prop.score_burst_from_blocks(dyn_noisy,curr_blocks,
                                                                   patchsize,nblocks)
        est.scores = scores
        est.scores_t = scores_t
        est.blocks = blocks

        # -- compute bootstrapping for the batch --
        eval_prop.score_fxn_name = "bs"
        eval_prop.score_cfg.bs_type = ""
        scores,scores_t,blocks = eval_plimb.score_burst_from_blocks(dyn_noisy,blocks,
                                                                    patchsize,nblocks)
        plimb.scores = scores
        plimb.scores_t = scores_t
        plimb.blocks = blocks


        # -- format results --
        pad = 3#2*(nframes-1)*ppf+4
        isize = edict({'h':H-pad,'w':W-pad})

        # -- flows to numpy --
        is_even = cfg.frame_size%2 == 0
        mid_pix = cfg.frame_size*cfg.frame_size//2 + (cfg.frame_size//2)*is_even
        mid_pix = 32*10+23
        flows_np = edict_torch_to_numpy(flows)

        # -- End-Point-Errors --
        epes_of = compute_flows_epe_wrt_ref(flows,"of")
        epes_nnf = compute_flows_epe_wrt_ref(flows,"nnf")
        epes_nnf_local = compute_flows_epe_wrt_ref(flows,"nnf_local")
        nnf_acc = compute_acc_wrt_ref(flows,"nnf")
        nnf_local_acc = compute_acc_wrt_ref(flows,"nnf_local")

        # -- PSNRs --
        aligned = remove_frame_centers(aligned)
        psnrs = compute_frames_psnr(aligned,isize)

        # -- print report ---
        print("\n"*3) # banner
        print("-"*25 + " Results " + "-"*25)
        print_dict_ndarray_0_midpix(flows_np,mid_pix)
        print_runtimes(runtimes)
        print_verbose_psnrs(psnrs)
        print_delta_summary_psnrs(psnrs)
        print_verbose_epes(epes_of,epes_nnf)
        print_nnf_acc(nnf_acc)
        print_nnf_local_acc(nnf_local_acc)
        print_summary_epes(epes_of,epes_nnf)
        print_summary_psnrs(psnrs)

        # -- prepare results to be appended --
        psnrs = edict_torch_to_numpy(psnrs)
        epes_of = edict_torch_to_numpy(epes_of)
        epes_nnf = edict_torch_to_numpy(epes_nnf)
        epes_nnf_local = edict_torch_to_numpy(epes_nnf_local)
        nnf_acc = edict_torch_to_numpy(nnf_acc)
        nnf_local_acc = edict_torch_to_numpy(nnf_local_acc)
        image_index  = torch_to_numpy(image_index)
        batch_results = {'runtimes':runtimes,
                         'optimal_scores':optimal_scores,
                         'psnrs':psnrs,
                         'epes_of':epes_of,
                         'epes_nnf':epes_nnf,
                         'epes_nnf_local':epes_nnf_local,
                         'nnf_acc':nnf_acc,
                         'nnf_local_acc':nnf_local_acc}
                         
        # -- format results --
        batch_results = flatten_internal_dict(batch_results)
        format_fields(batch_results,image_index,rng_state)

        print("shape check.")
        for key,value in batch_results.items():
            print(key,value.shape)

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
    for result_name,result_dict in results.items():
        for method_name,result in result_dict.items():
            method_dict[method_name][result_name] = result
            method_dict[method_name]['methods'] = method_name
        
    for key_a,value_a in method_dict.items():
        flattened.append(value_a)
    flattened = pd.DataFrame(flattened)
    dicts = flattened.to_dict('records')

    mgrouped = {}
    for elem in dicts:
        for key,value in elem.items():
            if key in mgrouped:
                mgrouped[key].append(value)
            else:
                mgrouped[key] = [value]

    return mgrouped


def format_fields(mgrouped,index,rng_state):

    # -- list keys --
    print(list(mgrouped.keys()))

    # -- get reference shapes --
    psnrs = mgrouped['psnrs']
    psnrs = np.stack(psnrs,axis=0)
    nmethods,nframes,batchsize = psnrs.shape

    # -- psnrs --
    psnrs = rearrange(psnrs,'m t i -> (m i) t')
    print("psnrs.shape: ",psnrs.shape)
    mgrouped['psnrs'] = psnrs

    # -- methods --
    methods = np.array(mgrouped['methods'])
    methods = repeat(methods,'m -> (m i)',i=batchsize)
    print("methods.shape: ",methods.shape)
    mgrouped['methods'] = methods

    # -- runtimes --
    runtimes = np.array(mgrouped['runtimes'])
    runtimes = repeat(runtimes,'m -> (m i)',i=batchsize)
    print("runtimes.shape: ",runtimes.shape)
    mgrouped['runtimes'] = runtimes

    # -- optimal scores --
    scores = np.array(mgrouped['optimal_scores'])
    scores = rearrange(scores,'m i p t -> (m i) p t',i=batchsize)
    print("scores.shape: ",scores.shape)
    mgrouped['optimal_scores'] = scores

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

    # -- epes_nnf_local --
    epes_nnf_local = np.array(mgrouped['epes_nnf_local'])
    epes_nnf_local = rearrange(epes_nnf_local,'m t i -> (m i) t')
    print("epes_nnf_local.shape: ",epes_nnf_local.shape)
    mgrouped['epes_nnf_local'] = epes_nnf_local


    # -- nnf_local_acc --
    print("NNF_LOCAL_ACC")
    nnf_local_acc = np.array(mgrouped['nnf_local_acc'])
    nnf_local_acc = rearrange(nnf_local_acc,'m t i -> (m i) t')
    print("nnf_local_acc.shape: ",nnf_local_acc.shape)
    mgrouped['nnf_local_acc'] = nnf_local_acc

    # -- nnf_acc --
    nnf_acc = np.array(mgrouped['nnf_acc'])
    nnf_acc = rearrange(nnf_acc,'m t i -> (m i) t')
    print("nnf_acc.shape: ",nnf_acc.shape)
    mgrouped['nnf_acc'] = nnf_acc

    # -- index --
    index = repeat(index,'i 1 -> (m i)',m=nmethods)
    print("index.shape: ",index.shape)    
    mgrouped['image_index'] = index

    # -- rng_state --
    rng_state = np.array([copy.deepcopy(rng_state) for m in range(nmethods)])
    print("rng_state.shape: ",rng_state.shape)
    mgrouped['rng_state'] = rng_state

    # -- test --
    df = pd.DataFrame().append(mgrouped,ignore_index=True)
    return df
