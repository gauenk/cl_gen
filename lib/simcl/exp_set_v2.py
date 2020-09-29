"""
prepare the "v1" set of experiments to run
"""

# python imports
import copy,uuid
from pathlib import Path

# project imports
import settings
from pyutils.deep_eq import deep_eq
from .config import get_args,get_cfg
from .exp_utils import check_exp_cache,load_exp_cache,save_exp_cache,get_record_experiment_path

def get_experiment_set_v2(rerun=False):
    version = 'v2'
    if check_exp_cache(version) and rerun is False:
        return load_exp_cache(version)

    # load default cfgs
    args = get_args()
    cfg = get_cfg(args)

    # determine experiment grid
    batch_grid = [1536,768,384,256]
    # batch_grid = [516,258,129,86]
    # batch_grid = [2*432,2*216,2*108,2*72]
    # init_lr_grid = [5e-4,1e-3]
    # init_lr_scale_grid = ['none','sqrt']
    # optimizer_grid = ['lars','adam']
    # sched_grid = ['lwca','rlop']
    # noise_params_grid = [{'stddev':50},{'stddev':100},{'stddev':10}]
    # hyper_parameters_grid = [{'h':0.},{'h':1.}]
    N_grid = [2,4,8,12]
    encoder_type_grid = ['resnet50','resnet18','simple']
    enc_size_grid = [1000,1000,768]

    # load the grid
    init_lr = 3e-1
    exps = []
    for bs,N in zip(batch_grid,N_grid):
        for encoder_type,enc_size in zip(encoder_type_grid,enc_size_grid):
            exp = _exp_from_cfg(cfg,bs,N,init_lr,encoder_type,enc_size)
            exps.append(exp)

    # save grid to cache
    save_exp_cache(exps,version)
    return exps        

def _exp_from_cfg(_cfg,bs,N,init_lr,encoder_type,enc_size):
    cfg = copy.deepcopy(_cfg)
    cfg.exp_name = str(uuid.uuid4())
    dsname = cfg.dataset.name.lower()
    cfg.model_path = Path(f"{settings.ROOT_PATH}/output/simcl/{dsname}/{cfg.exp_name}/model/")
    cfg.optim_path = Path(f"{settings.ROOT_PATH}/output/simcl/{dsname}/{cfg.exp_name}/optim/")
    cfg.summary_log_dir = Path(f"{settings.ROOT_PATH}/runs/simcl/{dsname}/{cfg.exp_name}/")

    cfg.noise_type = 'msg_simcl'
    cfg.noise_params['msg_simcl'] = {}
    cfg.noise_params['msg_simcl']['s'] = 0.5

    cfg.batch_size = bs
    cfg.N = N
    cfg.init_lr = init_lr
    cfg.encoder_type = encoder_type
    cfg.enc_size = enc_size
    return cfg
    
