
# -- python imports --
import time,os,copy,sys
import numpy as np
import pandas as pd
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- pytorch imports --
import torch

# -- faiss imports --
import faiss
sys.path.append("/home/gauenk/Documents/faiss/contrib/")
import nnf_utils as nnf_utils

# -- project imports --
import settings
import cache_io
from pyutils import tile_patches,save_image,torch_to_numpy,edict_torch_to_numpy
from pyutils.vst import anscombe
from patch_search import get_score_function
# from datasets.wrap_image_data import load_image_dataset,sample_to_cuda
from datasets import load_dataset,sample_to_cuda

# -- cuda profiler --
import nvtx

# -- [align] package imports --
import align.nnf as nnf
import align.nvof as nvof
from align.combo import EvalBlockScores,EvalBootBlockScores
from align.combo.optim import AlignOptimizer
from align.xforms import align_from_pix,flow_to_pix,create_isize,pix_to_flow,align_from_flow,flow_to_blocks
from align.interface import get_align_method


# -- [local] package imports --
from ._image_xforms import get_image_xform
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
    # data,loaders = load_image_dataset(cfg)
    data,loaders = load_dataset(cfg,cfg.dataset.mode)
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
    block_batchsize = 64
    eval_ave_simp = EvalBlockScores(score_fxn_ave,"ave",patchsize,block_batchsize,None)

    # -- create evaluator for ave --
    iterations,K = 1,1
    subsizes = []
    eval_ave = EvalBlockScores(score_fxn_ave,"ave",patchsize,block_batchsize,None)

    # -- create evaluator for bootstrapping --
    block_batchsize = 64
    eval_prop = EvalBlockScores(score_fxn_bs,"bs",patchsize,block_batchsize,None)

    # -- init flownet model --
    cfg.gpuid = 1 - cfg.gpuid # flip. flop.
    flownet_align = get_align_method(cfg,"flownet_v2",False)
    cfg.gpuid = 1 - cfg.gpuid # flippity flop.

    # -- get an image transform --
    image_xform = get_image_xform(cfg.image_xform,cfg.gpuid,cfg.frame_size)
    blockLabels,_ = nnf_utils.getBlockLabels(None,nblocks,np.int32,cfg.device,True)
    
    # -- iterate over images --
    for image_bindex in range(NUM_BATCHES):

        print("-="*30+"-")
        print(f"Running image batch index: {image_bindex}")
        print("-="*30+"-")
        torch.cuda.empty_cache()

        # -- sample & unpack batch --
        sample = next(image_iter)
        sample_to_cuda(sample)
        convert_keys(sample)
        torch.cuda.synchronize()
        # for key,val in sample.items():
        #     print(key,type(val))
        #     if torch.is_tensor(val):
        #         print(key,val.device)

        dyn_noisy = sample['dyn_noisy'] # dynamics and noise
        dyn_clean = sample['dyn_clean'] # dynamics and no noise
        static_noisy = sample['static_noisy'] # no dynamics and noise
        static_clean = sample['static_clean'] # no dynamics and no noise
        nnf_gt = sample['nnf']
        if nnf_gt.ndim == 6:
            nnf_gt = nnf_gt[:,:,0] # pick top 1 out of K
        pix_gt = nnf_gt.type(torch.float)
        image_index = sample['image_index']
        rng_state = sample['rng_state']

        # TODO: anscombe is a type of image transform
        if not(image_xform is None):
            dyn_clean_ftrs = image_xform(dyn_clean)
            dyn_noisy_ftrs = image_xform(dyn_noisy)
        else:
            dyn_clean_ftrs = dyn_clean
            dyn_noisy_ftrs = dyn_noisy

        # print("SHAPES")
        # print(dyn_noisy.shape)
        # print(dyn_clean.shape)
        # print(nnf_gt.shape)

        # -- shape info --
        pad = cfg.nblocks//2+1
        T,B,C,H,W = dyn_noisy.shape
        isize = edict({'h':H,'w':W})
        psize = edict({'h':H-2*pad,'w':W-2*pad})
        ref_t = nframes//2
        nimages,npix,nframes = B,H*W,T

        # -- create results dict --
        pixs = edict()
        flows = edict()
        aligned = edict()
        runtimes = edict()
        optimal_scores = edict() # score function at optimal


        # -- groundtruth flow --
        if pix_gt.ndim == 3:
            pix_gt_rs = rearrange(pix_gt,'i tm1 two -> i 1 tm1 two')
            pix_gt = repeat(pix_gt,'i tm1 two -> i p tm1 two',p=npix)
        if pix_gt.ndim == 5:
            pix_gt = rearrange(pix_gt,'t i h w two -> i (h w) t two')
        pix_gt = torch.LongTensor(pix_gt.cpu().numpy().copy())
        flows.of = torch.zeros_like(pix_gt)#pix_to_flow(pix_gt.clone())
        aligned.of = align_from_flow(dyn_clean,flows.of,nblocks,isize=isize)
        pixs.of = flow_to_pix(flows.of.clone(),nframes,isize=isize)
        runtimes.of = 0. # given
        optimal_scores.of = np.zeros((nimages,npix,1,nframes)) # clean target is zero
        aligned.clean = static_clean
        # optimal_scores.of = eval_ave.score_burst_from_flow(dyn_noisy,
        #                                                    flows.of,
        #                                                    patchsize,nblocks)[0]

        # -- compute nearest neighbor fields [global] --
        print("NNF Global.")
        start_time = time.perf_counter()
        shape_str = 't b h w two -> b (h w) t two'
        nnf_vals,nnf_pix = nnf.compute_burst_nnf(dyn_clean_ftrs,ref_t,patchsize)
        runtimes.nnf = time.perf_counter() - start_time
        pixs.nnf = torch.LongTensor(rearrange(nnf_pix[...,0,:],shape_str))
        flows.nnf = pix_to_flow(pixs.nnf.clone())
        aligned.nnf = align_from_pix(dyn_clean,pixs.nnf,nblocks)
        # aligned.nnf = align_from_flow(dyn_clean,flows.nnf,nblocks,isize=isize)
        optimal_scores.nnf = np.zeros((nimages,npix,1,nframes)) # clean target is zero

        # -- compute nearest neighbor fields [local] --
        print("NNF Local.")
        start_time = time.perf_counter()
        local_vals,local_locs = nnf_vals,nnf_pix
        # local_vals,local_locs = nnf_utils.runNnfBurst(dyn_clean_ftrs,
        #                                                   patchsize, nblocks,
        #                                                   1, valMean = 0.,
        #                                                   blockLabels=blockLabels)
        runtimes.nnf_local = time.perf_counter() - start_time
        # pixs.nnf_local = rearrange(local_locs[...,0,:],'t i h w two -> i (h w) t two')
        # pixs.nnf_local = torch.LongTensor(pixs.nnf_local)
        # pixs.nnf_local = torch.LongTensor(rearrange(nnf_pix[...,0,:],shape_str))
        pixs.nnf_local = pixs.nnf.clone()
        flows.nnf_local = pix_to_flow(pixs.nnf_local.clone())
        print("flows.nnf_local.shape ",flows.nnf_local.shape)
        print(flows.nnf_local.min(),flows.nnf_local.max())


        # aligned.nnf_local = align_from_pix(dyn_clean,pixs.nnf_local,nblocks)
        # optimal_scores.nnf_local = eval_ave.score_burst_from_flow(dyn_noisy,
        #                                                           flows.nnf_local,
        #                                                           patchsize,nblocks)[1]
        aligned.nnf_local = align_from_pix(dyn_clean,pixs.nnf,nblocks)
        optimal_scores.nnf_local = eval_ave.score_burst_from_flow(dyn_noisy,
                                                                  flows.nnf,
                                                                  patchsize,nblocks)[1]
        optimal_scores.nnf_local = torch_to_numpy(optimal_scores.nnf_local)

        # -----------------------------------
        #
        # -- old way to compute NNF local --
        # 
        # -----------------------------------

        # pixs.nnf = torch.LongTensor(rearrange(nnf_pix[...,0,:],shape_str))
        # flows.nnf = pix_to_flow(pixs.nnf.clone())
        # aligned.nnf = align_from_pix(dyn_clean,pixs.nnf,nblocks)
        # aligned.nnf = align_from_flow(dyn_clean,flows.nnf,nblocks,isize=isize)

        # flows.nnf_local = optim.run(dyn_clean_ftrs,patchsize,eval_ave,
        #                             nblocks,iterations,subsizes,K)

        # -----------------------------------
        # -----------------------------------


        # -- compute proposed search of nnf --
        print("Global NNF Noisy")
        start_time = time.perf_counter()
        split_vals,split_pix = nnf.compute_burst_nnf(dyn_noisy_ftrs,ref_t,patchsize)
        runtimes.split = time.perf_counter() - start_time
        # split_pix = np.copy(nnf_pix)
        split_pix_best = torch.LongTensor(rearrange(split_pix[...,0,:],shape_str))
        split_pix_best = torch.LongTensor(split_pix_best)
        pixs.split = split_pix_best.clone()
        flows.split = pix_to_flow(split_pix_best)
        aligned.split = align_from_pix(dyn_clean,split_pix_best,nblocks)
        optimal_scores.split = eval_ave.score_burst_from_flow(dyn_noisy,flows.nnf_local,
                                                              patchsize,nblocks)[1]
        optimal_scores.split = torch_to_numpy(optimal_scores.split)

        # -- compute complex ave --
        iterations,K = 0,1
        subsizes = []
        print("[complex] Ave loss function")
        start_time = time.perf_counter()
        optim = AlignOptimizer("v3")
        flows.ave = optim.run(dyn_noisy_ftrs,patchsize,eval_ave,
                             nblocks,iterations,subsizes,K)
        # flows.ave = flows.of.clone()
        runtimes.ave = time.perf_counter() - start_time
        pixs.ave = flow_to_pix(flows.ave.clone(),nframes,isize=isize)
        aligned.ave = align_from_flow(dyn_clean,flows.ave,nblocks,isize=isize)
        optimal_scores.ave = optimal_scores.split # same "ave" function

        # -- compute nvof flow --
        print("NVOF")
        start_time = time.perf_counter()
        flows.nvof = nvof.nvof_burst(dyn_noisy_ftrs)
        runtimes.nvof = time.perf_counter() - start_time
        pixs.nvof = flow_to_pix(flows.nvof.clone(),nframes,isize=isize)
        aligned.nvof = align_from_flow(dyn_clean,flows.nvof,nblocks,isize=isize)
        optimal_scores.nvof = optimal_scores.split # same "ave" function

        # -- compute flownet --
        print("FlowNetv2")
        start_time = time.perf_counter()
        _,flows.flownet = flownet_align(dyn_noisy_ftrs)
        runtimes.flownet = time.perf_counter() - start_time
        pixs.flownet = flow_to_pix(flows.flownet.clone(),nframes,isize=isize)
        aligned.flownet = align_from_flow(dyn_clean,flows.flownet,nblocks,isize=isize)
        optimal_scores.flownet = optimal_scores.split

        # -- compute simple ave --
        iterations,K = 0,1
        subsizes = []
        print("[simple] Ave loss function")
        start_time = time.perf_counter()
        optim = AlignOptimizer("v3")
        # flows.ave_simp = optim.run(dyn_noisy,patchsize,eval_ave_simp,
        #                      nblocks,iterations,subsizes,K)
        flows.ave_simp = flows.ave.clone().cpu()
        pixs.ave_simp = flow_to_pix(flows.ave_simp.clone(),nframes,isize=isize)
        aligned.ave_simp = align_from_flow(dyn_clean,flows.ave_simp,nblocks,isize=isize)
        runtimes.ave_simp = runtimes.ave#time.perf_counter() - start_time
        optimal_scores.ave_simp = optimal_scores.split # same "ave" function

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
            subsizes = [2,]*cfg.nframes*3
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
        # flows.est = optim.run(dyn_noisy_ftrs,patchsize,eval_prop,
        #                      nblocks,iterations,subsizes,K)
        flows.est = flows.of.clone()
        pixs.est = flow_to_pix(flows.est.clone(),nframes,isize=isize)
        aligned.est = align_from_flow(dyn_clean,flows.est,patchsize,isize=isize)
        runtimes.est = time.perf_counter() - start_time
        optimal_scores.est = eval_prop.score_burst_from_flow(dyn_noisy,flows.nnf_local,
                                                             patchsize,nblocks)[1]
        optimal_scores.est = torch_to_numpy(optimal_scores.est)

        # aligned.est = aligned.of.clone()
        # time_est = 0.

        # -- format results --
        #pad = 2*(nframes-1)*ppf+4
        # pad = 2*(cfg.nblocks//2)#2*(nframes-1)*ppf+4
        # isize = edict({'h':H-pad,'w':W-pad})

        # -- flows to numpy --
        frame_size = cfg.frame_size[0]
        is_even = frame_size%2 == 0
        mid_pix = frame_size*frame_size//2 + (frame_size//2)*is_even
        mid_pix = 32*10+23
        flows_np = edict_torch_to_numpy(flows)
        pixs_np = edict_torch_to_numpy(pixs)

        # -- End-Point-Errors --
        epes_of = compute_flows_epe_wrt_ref(flows,"of")
        epes_nnf = compute_flows_epe_wrt_ref(flows,"nnf")
        epes_nnf_local = compute_flows_epe_wrt_ref(flows,"nnf_local")
        nnf_acc = compute_acc_wrt_ref(flows,"nnf")
        nnf_local_acc = compute_acc_wrt_ref(flows,"nnf_local")

        # -- PSNRs --
        aligned = remove_frame_centers(aligned)
        psnrs = compute_frames_psnr(aligned,psize)

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
    scores = rearrange(scores,'m i p 1 t -> (m i) p t',i=batchsize)
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
