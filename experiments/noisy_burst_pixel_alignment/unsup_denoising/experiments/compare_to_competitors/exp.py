
# -- python imports --
import time,sys,os
import numpy as np
import pandas as pd
from pathlib import Path
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- multiprocessing --
from multiprocessing import current_process

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

# -- [local] package imports --
from .exp_utils import *
from .learn import train_model,test_model
from ._sim_methods import get_sim_method
from ._nn_archs import get_nn_model

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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

    # -- reset sys.out if subprocess --
    cproc = current_process()
    if not(cfg.pid == cproc.pid):
        printfn = Path("./running")  / f"{os.getpid()}.txt"
        orig_stdout = sys.stdout
        f = open(printfn, 'w')
        sys.stdout = f

    # -- init exp! --
    print("RUNNING Exp: [UNSUP DENOISING] Compare to Competitors")
    print(cfg)

    # -- set default device --
    torch.cuda.set_device(cfg.gpuid)

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

    # -- get neural netowrk --
    model,loss_fxn,optim,sched_fxn = get_nn_model(cfg,cfg.nn_arch)

    # -- get sim method --
    sim_fxn = get_sim_method(cfg,cfg.sim_method)
    
    # -- some constants --
    nframes,nblocks = cfg.nframes,cfg.nblocks 
    patchsize = cfg.patchsize
    ppf = cfg.dynamic_info.ppf
    check_parameters(nblocks,patchsize)

    # -- iterate over images --
    start_time = time.perf_counter()
    results = {}
    for epoch in range(cfg.nepochs):
        print("-"*25)
        print(f"Epoch [{epoch}]")
        print("-"*25)
        sched_fxn(epoch)
        result_tr = train_model(cfg,model,loss_fxn,optim,loaders.tr,sim_fxn)
        append_result_to_dict(results,result_tr)
        if epoch % cfg.test_interval == 0:
            result_te = test_model(cfg,model,loaders.te,loss_fxn,epoch)
            append_result_to_dict(results,result_te)
        if epoch % cfg.save_interval == 0: pass
    result_te = test_model(cfg,model,loaders.te,loss_fxn,epoch)
    append_result_to_dict(results,result_te)
    runtime = time.perf_counter() - start_time

    # -- format results --
    # listdict_to_numpy(results)
    results['runtime'] = np.array([runtime])

    return results

def listdict_to_numpy(adict):
    for key,val in adict.items():
        print(key,type(val))
        if isinstance(val,list):
            print(type(val[0]))
            if isinstance(val[0],list):
                adict[key] = np.array(val)
            elif isinstance(val[0],np.ndarray):
                adict[key] = np.stack(val)
            elif torch.is_tensor(val[0]):
                adict[key] = torch.stack(val).numpy()
            else:
                adict[key] = np.array(val)                
    return adict

def append_result_to_dict(records,epoch_result):
    for rdict in epoch_result:
        for key,val in rdict.items():
            if isinstance(val,list) and len(val) > 0:
                if isinstance(val[0],list):
                    val = np.array(val)
                elif isinstance(val[0],np.ndarray):
                    val = np.stack(val)
                elif torch.is_tensor(val[0]):
                    val = torch.stack(val).numpy()
                else:
                    val = np.array(val)
            elif torch.is_tensor(val):
                    val = val.numpy()
            if key in records: records[key].append(val)
            else: records[key] = [val]

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
