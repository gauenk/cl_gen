
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
import bnnf_utils as bnnf_utils

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
import align.burstNnf as burstNnf
import align.cflow as cflow
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
    ps = patchsize
    ppf = cfg.dynamic_info.ppf
    check_parameters(nblocks,patchsize)

    # -- create evaluator for ave; simple --
    iterations,K = 1,1
    subsizes = []
    block_batchsize = 32
    eval_ave_simp = EvalBlockScores(score_fxn_ave,"ave",patchsize,block_batchsize,None)

    # -- create evaluator for ave --
    iterations,K = 1,1
    subsizes = []
    eval_ave = EvalBlockScores(score_fxn_ave,"ave",patchsize,block_batchsize,None)

    # -- create evaluator for bootstrapping --
    block_batchsize = 32
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
        flow_gt = sample['flow'] 
        if nnf_gt.ndim == 6:
            nnf_gt = nnf_gt[:,:,0] # pick top 1 out of K
        image_index = sample['image_index']
        rng_state = sample['rng_state']

        # TODO: anscombe is a type of image transform
        if not(image_xform is None):
            dyn_clean_ftrs = image_xform(dyn_clean)
            dyn_noisy_ftrs = image_xform(dyn_noisy)
        else:
            dyn_clean_ftrs = dyn_clean
            dyn_noisy_ftrs = dyn_noisy
        
        if "resize" in cfg.image_xform:
            print("Images, Flows, and NNF Modified.")
            dyn_clean = image_xform(dyn_clean)
            dyn_noisy = image_xform(dyn_noisy)
            T,B,C,H,W = dyn_noisy.shape
            flow_gt = torch.zeros((B,T,H,W,2))
            nnf_gt = torch.zeros((T,B,H,W,2))


        save_image(dyn_clean,"dyn_clean.png")
        # print("SHAPES")
        # print(dyn_noisy.shape)
        # print(dyn_clean.shape)
        # print(nnf_gt.shape)

        # -- shape info --
        pad = cfg.nblocks//2+cfg.patchsize//2
        T,B,C,H,W = dyn_noisy.shape
        isize = edict({'h':H,'w':W})
        psize = edict({'h':H-2*pad,'w':W-2*pad})
        ref_t = nframes//2
        nimages,npix,nframes = B,H*W,T
        frame_size = [H,W]

        # -- create results dict --
        pixs = edict()
        flows = edict()
        anoisy = edict()
        aligned = edict()
        runtimes = edict()
        optimal_scores = edict() # score function at optimal

        # -- compute proposed search of nnf --
        # ave = torch.mean(dyn_noisy_ftrs[:,0,:,4:4+ps,4:4+ps],dim=0)
        # frames = dyn_noisy_ftrs[:,0,:,4:4+ps,4:4+ps]
        # gt_offset = torch.sum((frames - ave)**2/nframes).item()
        # print("Optimal: ",gt_offset)
        gt_offset = -1.

        # -- compute proposed search of nnf --
        print("Our Method")
        # flow_fmt = rearrange(flow_gt,'i t h w two -> i (h w) t two')
        # mode = bnnf_utils.evalAtFlow(dyn_noisy_ftrs, flow_fmt, patchsize,
        #                              nblocks, return_mode=True)
        # print("mode: ",mode)
        print("dyn_noisy_ftrs.shape ",dyn_noisy_ftrs.shape)
        valMean = 0.#mode
        print("valMean: ",valMean)
        start_time = time.perf_counter()
        _,flows.est = bnnf_utils.runBurstNnf(dyn_noisy_ftrs, patchsize,
                                             nblocks, k = 1,
                                             valMean = valMean, blockLabels=None,
                                             fmt = True, to_flow=True)
        flows.est = flows.est[0]        
        # flows.est = rearrange(flow_gt,'i t h w two -> i (h w) t two')
        runtimes.est = time.perf_counter() - start_time
        pixs.est = flow_to_pix(flows.est.clone(),nframes,isize=isize)
        aligned.est = align_from_flow(dyn_clean,flows.est,patchsize,isize=isize)
        anoisy.est = align_from_flow(dyn_noisy,flows.est,patchsize,isize=isize)
        optimal_scores.est = np.zeros((nimages,npix,1,nframes))

        # -- compute proposed search of nnf [with tiling ]--
        print("Our Method")
        # flow_fmt = rearrange(flow_gt,'i t h w two -> i (h w) t two')
        # mode = bnnf_utils.evalAtFlow(dyn_noisy_ftrs, flow_fmt, patchsize,
        #                              nblocks, return_mode=True)
        # print("mode: ",mode)
        print("dyn_noisy_ftrs.shape ",dyn_noisy_ftrs.shape)
        valMean = 0.#mode
        print("valMean: ",valMean)
        start_time = time.perf_counter()
        _,flows.est_tile = bnnf_utils.runBurstNnf(dyn_noisy_ftrs, patchsize,
                                             nblocks, k = 1,
                                             valMean = valMean, blockLabels=None,
                                             fmt = True, to_flow=True,
                                             tile_burst=False)
        flows.est_tile = flows.est_tile[0]        
        # flows.est_tile = rearrange(flow_gt,'i t h w two -> i (h w) t two')
        runtimes.est_tile = time.perf_counter() - start_time
        pixs.est_tile = flow_to_pix(flows.est_tile.clone(),nframes,isize=isize)
        aligned.est_tile = align_from_flow(dyn_clean,flows.est_tile,patchsize,isize=isize)
        anoisy.est_tile = align_from_flow(dyn_noisy,flows.est_tile,patchsize,isize=isize)
        optimal_scores.est_tile = np.zeros((nimages,npix,1,nframes))

        # -- compute new est method --
        print("[Burst-LK] loss function")
        print(flow_gt.shape)
        print(flow_gt[0,:3,32,32,:])
        print(flow_gt.shape)
        start_time = time.perf_counter()
        if frame_size[0] <= 64 and cfg.nblocks < 10 and True:
            flows.blk = burstNnf.run(dyn_noisy_ftrs,patchsize,nblocks)
        else:
            flows.blk = rearrange(flow_gt,'i t h w two -> i (h w) t two')
        runtimes.blk = time.perf_counter() - start_time
        pixs.blk = flow_to_pix(flows.blk.clone(),nframes,isize=isize)
        aligned.blk = align_from_flow(dyn_clean,flows.blk,patchsize,isize=isize)
        optimal_scores.blk = np.zeros((nimages,npix,1,nframes))
        # optimal_scores.blk = eval_prop.score_burst_from_flow(dyn_noisy,flows.nnf_local,
        #                                                      patchsize,nblocks)[1]
        optimal_scores.blk = torch_to_numpy(optimal_scores.blk)

        # -- compute optical flow --
        print("[C Flow]")
        print(dyn_noisy_ftrs.shape)
        start_time = time.perf_counter()
        # flows.cflow = cflow.runBurst(dyn_clean_ftrs)
        # flows.cflow[...,1] = -flows.cflow[...,1]
        flows.cflow = torch.LongTensor(flows.blk.clone().cpu().numpy())
        # flows.cflow = flows.blk.clone()
        # flows.cflow = torch.round(flows.cflow)
        runtimes.cflow = time.perf_counter() - start_time
        pixs.cflow = flow_to_pix(flows.cflow.clone(),nframes,isize=isize)
        aligned.cflow = align_from_flow(dyn_clean,flows.cflow,patchsize,isize=isize)
        optimal_scores.cflow = np.zeros((nimages,npix,1,nframes))
        # optimal_scores.blk = eval_prop.score_burst_from_flow(dyn_noisy,flows.nnf_local,
        #                                                      patchsize,nblocks)[1]
        optimal_scores.blk = torch_to_numpy(optimal_scores.blk)

        # -- compute groundtruth flow --
        dsname = cfg.dataset.name
        if "kitti" in dsname or 'bsd_burst' == dsname:
            pix_gt = nnf_gt.type(torch.float)
            if pix_gt.ndim == 3:
                pix_gt_rs = rearrange(pix_gt,'i tm1 two -> i 1 tm1 two')
                pix_gt = repeat(pix_gt,'i tm1 two -> i p tm1 two',p=npix)
            if pix_gt.ndim == 5:
                pix_gt = rearrange(pix_gt,'t i h w two -> i (h w) t two')
            pix_gt = torch.LongTensor(pix_gt.cpu().numpy().copy())
            # flows.of = torch.zeros_like(pix_gt)#pix_to_flow(pix_gt.clone())
            flows.of = pix_to_flow(pix_gt.clone())
        else:
            flows.of = flow_gt
            flows.of = rearrange(flow_gt,'i t h w two -> i (h w) t two')
        # -- align groundtruth flow --
        aligned.of = align_from_flow(dyn_clean,flows.of,nblocks,isize=isize)
        pixs.of = flow_to_pix(flows.of.clone(),nframes,isize=isize)
        runtimes.of = 0. # given
        optimal_scores.of = np.zeros((nimages,npix,1,nframes)) # clean target is zero
        aligned.clean = static_clean
        anoisy.clean = static_clean
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
        print(dyn_clean.shape,pixs.nnf.shape,nblocks)
        aligned.nnf = align_from_pix(dyn_clean,pixs.nnf,nblocks)
        anoisy.nnf = align_from_pix(dyn_noisy,pixs.nnf,nblocks)
        # aligned.nnf = align_from_flow(dyn_clean,flows.nnf,nblocks,isize=isize)
        optimal_scores.nnf = np.zeros((nimages,npix,1,nframes)) # clean target is zero

        # -- compute nearest neighbor fields [local] --
        print("NNF Local.")
        start_time = time.perf_counter()
        valMean = 0.
        vals_local,pix_local = nnf_utils.runNnfBurst(dyn_clean_ftrs,
                                                      patchsize, nblocks,
                                                      1, valMean = valMean,
                                                      blockLabels=blockLabels)
        runtimes.nnf_local = time.perf_counter() - start_time
        torch.cuda.synchronize()
        print("pix_local.shape ",pix_local.shape)
        pixs.nnf_local = rearrange(pix_local,'t i h w 1 two -> i (h w) t two')
        flows.nnf_local = pix_to_flow(pixs.nnf_local.clone())
        # aligned_local = align_from_flow(clean,flow_gt,cfg.nblocks)
        # aligned_local = align_from_pix(dyn_clean,pix_local,cfg.nblocks)
        print(flows.nnf_local.min(),flows.nnf_local.max())
        aligned.nnf_local = align_from_pix(dyn_clean,pixs.nnf_local,nblocks)
        anoisy.nnf_local = align_from_pix(dyn_noisy,pixs.nnf_local,nblocks)
        optimal_scores.nnf_local = optimal_scores.nnf
        # optimal_scores.nnf_local = eval_ave.score_burst_from_flow(dyn_noisy,
        #                                                           flows.nnf,
        #                                                           patchsize,nblocks)[1]
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
        anoisy.split = align_from_pix(dyn_noisy,split_pix_best,nblocks)
        optimal_scores.split = optimal_scores.nnf_local
        # optimal_scores.split = eval_ave.score_burst_from_flow(dyn_noisy,flows.nnf_local,
        #                                                       patchsize,nblocks)[1]
        optimal_scores.split = torch_to_numpy(optimal_scores.split)

        # -- compute complex ave --
        iterations,K = 0,1
        subsizes = []
        print("[Ours] Ave loss function")
        start_time = time.perf_counter()
        estVar = torch.std(dyn_noisy_ftrs.reshape(-1)).item()**2
        valMean = 2 * estVar# * patchsize**2# / patchsize**2
        vals_local,pix_local = nnf_utils.runNnfBurst(dyn_noisy_ftrs,
                                                     patchsize, nblocks,
                                                     1, valMean = valMean,
                                                     blockLabels=blockLabels)
        runtimes.ave = time.perf_counter() - start_time
        pixs.ave = rearrange(pix_local,'t i h w 1 two -> i (h w) t two')
        flows.ave = pix_to_flow(pixs.ave.clone())
        optimal_scores.ave = optimal_scores.split # same "ave" function
        aligned.ave = align_from_flow(dyn_clean,flows.ave,nblocks,isize=isize)
        anoisy.ave = align_from_flow(dyn_noisy,flows.ave,nblocks,isize=isize)
        optimal_scores.ave = optimal_scores.split # same "ave" function

        # -- compute  flow --
        print("L2-Local Recursive")
        start_time = time.perf_counter()
        vals_local,pix_local,wburst = nnf_utils.runNnfBurstRecursive(dyn_noisy_ftrs,
                                                                     dyn_clean,
                                                                     patchsize, nblocks,
                                                                     isize, 1,
                                                                     valMean = valMean,
                                                                     blockLabels=
                                                                     blockLabels)
        runtimes.l2r = time.perf_counter() - start_time
        pixs.l2r = rearrange(pix_local,'t i h w 1 two -> i (h w) t two')
        flows.l2r = pix_to_flow(pixs.l2r.clone())
        aligned.l2r = wburst#align_from_flow(dyn_clean,flows.l2r,nblocks,isize=isize)
        optimal_scores.l2r = optimal_scores.split # same "ave" function

        # -- compute nvof flow --
        print("NVOF")
        start_time = time.perf_counter()
        # flows.nvof = nvof.nvof_burst(dyn_noisy_ftrs)
        flows.nvof = flows.ave.clone()
        runtimes.nvof = time.perf_counter() - start_time
        pixs.nvof = flow_to_pix(flows.nvof.clone(),nframes,isize=isize)
        aligned.nvof = align_from_flow(dyn_clean,flows.nvof,nblocks,isize=isize)
        anoisy.nvof = align_from_flow(dyn_noisy,flows.nvof,nblocks,isize=isize)
        optimal_scores.nvof = optimal_scores.split # same "ave" function

        # -- compute flownet --
        print("FlowNetv2")
        start_time = time.perf_counter()
        _,flows.flownet = flownet_align(dyn_noisy_ftrs)
        runtimes.flownet = time.perf_counter() - start_time
        pixs.flownet = flow_to_pix(flows.flownet.clone(),nframes,isize=isize)
        aligned.flownet = align_from_flow(dyn_clean,flows.flownet,nblocks,isize=isize)
        anoisy.flownet = align_from_flow(dyn_noisy,flows.flownet,nblocks,isize=isize)
        optimal_scores.flownet = optimal_scores.split

        # -- compute simple ave --
        iterations,K = 0,1
        subsizes = []
        print("[simple] Ave loss function")
        start_time = time.perf_counter()
        optim = AlignOptimizer("v3")
        if cfg.patchsize < 11 and cfg.frame_size[0] <= 64 and False:
            flows.ave_simp = optim.run(dyn_noisy,patchsize,eval_ave_simp,
                                       nblocks,iterations,subsizes,K)
        else:
            flows.ave_simp = flows.ave.clone().cpu()
        runtimes.ave_simp = time.perf_counter() - start_time
        pixs.ave_simp = flow_to_pix(flows.ave_simp.clone(),nframes,isize=isize)
        aligned.ave_simp = align_from_flow(dyn_clean,flows.ave_simp,nblocks,isize=isize)
        anoisy.ave_simp = align_from_flow(dyn_noisy,flows.ave_simp,nblocks,isize=isize)
        optimal_scores.ave_simp = optimal_scores.split # same "ave" function

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
        aligned = remove_center_frames(aligned)
        psnrs = compute_frames_psnr(aligned,psize)

        # -- denoised PSNRS --
        def burst_mean(in_burst): return torch.mean(in_burst,dim=0)[None,:]
        anoisy = remove_center_frames(anoisy)
        anoisy = apply_across_dict(anoisy,burst_mean)
        dn_psnrs = compute_frames_psnr(anoisy,psize)
        print(dn_psnrs)

        # -- print report ---
        print("\n"*3) # banner
        print("-"*25 + " Results " + "-"*25)
        print_dict_ndarray_0_midpix(flows_np,mid_pix)
        print_dict_ndarray_0_midpix(pixs_np,mid_pix)
        print_runtimes(runtimes)
        # print_verbose_psnrs(psnrs)
        # print_delta_summary_psnrs(psnrs)
        # print_verbose_epes(epes_of,epes_nnf)
        # print_nnf_acc(nnf_acc)
        # print_nnf_local_acc(nnf_local_acc)
        # print_summary_epes(epes_of,epes_nnf)
        print_summary_denoised_psnrs(dn_psnrs)
        print_summary_psnrs(psnrs)
        print_runtimes(runtimes)


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


        # print("shape check.")
        # for key,value in batch_results.items():
        #     print(key,value.shape)

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
    # print(list(mgrouped.keys()))

    # -- get reference shapes --
    psnrs = mgrouped['psnrs']
    psnrs = np.stack(psnrs,axis=0)
    nmethods,nframes,batchsize = psnrs.shape

    # -- psnrs --
    psnrs = rearrange(psnrs,'m t i -> (m i) t')
    # print("psnrs.shape: ",psnrs.shape)
    mgrouped['psnrs'] = psnrs

    # -- methods --
    methods = np.array(mgrouped['methods'])
    methods = repeat(methods,'m -> (m i)',i=batchsize)
    # print("methods.shape: ",methods.shape)
    mgrouped['methods'] = methods

    # -- runtimes --
    runtimes = np.array(mgrouped['runtimes'])
    runtimes = repeat(runtimes,'m -> (m i)',i=batchsize)
    # print("runtimes.shape: ",runtimes.shape)
    mgrouped['runtimes'] = runtimes

    # -- optimal scores --
    scores = np.array(mgrouped['optimal_scores'])
    scores = rearrange(scores,'m i p 1 t -> (m i) p t',i=batchsize)
    # print("scores.shape: ",scores.shape)
    mgrouped['optimal_scores'] = scores

    # -- epes_of --
    epes_of = np.array(mgrouped['epes_of'])
    epes_of = rearrange(epes_of,'m t i -> (m i) t')
    # print("epes_of.shape: ",epes_of.shape)
    mgrouped['epes_of'] = epes_of

    # -- epes_nnf --
    epes_nnf = np.array(mgrouped['epes_nnf'])
    epes_nnf = rearrange(epes_nnf,'m t i -> (m i) t')
    # print("epes_nnf.shape: ",epes_nnf.shape)
    mgrouped['epes_nnf'] = epes_nnf

    # -- epes_nnf_local --
    epes_nnf_local = np.array(mgrouped['epes_nnf_local'])
    epes_nnf_local = rearrange(epes_nnf_local,'m t i -> (m i) t')
    # print("epes_nnf_local.shape: ",epes_nnf_local.shape)
    mgrouped['epes_nnf_local'] = epes_nnf_local


    # -- nnf_local_acc --
    # print("NNF_LOCAL_ACC")
    nnf_local_acc = np.array(mgrouped['nnf_local_acc'])
    nnf_local_acc = rearrange(nnf_local_acc,'m t i -> (m i) t')
    # print("nnf_local_acc.shape: ",nnf_local_acc.shape)
    mgrouped['nnf_local_acc'] = nnf_local_acc

    # -- nnf_acc --
    nnf_acc = np.array(mgrouped['nnf_acc'])
    nnf_acc = rearrange(nnf_acc,'m t i -> (m i) t')
    # print("nnf_acc.shape: ",nnf_acc.shape)
    mgrouped['nnf_acc'] = nnf_acc

    # -- index --
    index = repeat(index,'i 1 -> (m i)',m=nmethods)
    # print("index.shape: ",index.shape)    
    mgrouped['image_index'] = index

    # -- rng_state --
    rng_state = np.array([copy.deepcopy(rng_state) for m in range(nmethods)])
    # print("rng_state.shape: ",rng_state.shape)
    mgrouped['rng_state'] = rng_state

    # -- test --
    df = pd.DataFrame().append(mgrouped,ignore_index=True)
    return df
