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

def test_get_experiment_set_v1():
    # test has passed!
    a = get_experiment_set_v1(rerun=True)
    b = get_experiment_set_v1(rerun=False)
    return deep_eq(a,b)

def get_experiment_set_v1(rerun=False):
    version = 'v1'
    if check_exp_cache(version) and rerun is False:
        return load_exp_cache(version)

    # load default cfgs
    args = get_args()
    cfg = get_cfg(args)

    # determine experiment grid
    batch_grid = [1536,768,384,256]
    # batch_grid = [516,258,129,86]
    # batch_grid = [2*432,2*216,2*108,2*72]
    N_grid = [2,4,8,12]
    init_lr_grid = [5e-4,1e-3]
    init_lr_scale_grid = ['none','sqrt']
    optimizer_grid = ['adam','lars']
    sched_grid = ['rlop','lwca']
    hyper_parameters_grid = [{'h':0.},{'h':1.}]
    noise_params_grid = [{'stddev':50},{'stddev':100},{'stddev':10}]

    # load the grid
    init_lr = 1e-2
    exps = []
    for bs,N in zip(batch_grid,N_grid):
        for hyper in hyper_parameters_grid:
            for noise_params in noise_params_grid:
                exp = _exp_from_cfg(cfg,noise_params,bs,N,hyper,init_lr)
                exps.append(exp)

    # save grid to cache
    save_exp_cache(exps,version)
    return exps        

def _exp_from_cfg(_cfg,noise_params,bs,N,hyper,init_lr):
    cfg = copy.deepcopy(_cfg)
    cfg.exp_name = str(uuid.uuid4())
    dsname = cfg.dataset.name.lower()
    cfg.model_path = Path(f"{settings.ROOT_PATH}/output/simcl/{dsname}/{cfg.exp_name}/model/")
    cfg.optim_path = Path(f"{settings.ROOT_PATH}/output/simcl/{dsname}/{cfg.exp_name}/optim/")
    cfg.summary_log_dir = Path(f"{settings.ROOT_PATH}/runs/simcl/{dsname}/{cfg.exp_name}/")
    cfg.init_lr = init_lr
    cfg.noise_type = 'msg_simcl'
    cfg.noise_params['msg_simcl'] = {}
    cfg.noise_params['msg_simcl']['s'] = 1.0
    cfg.noise_params['g'] = noise_params
    cfg.noise_params['g']['mean'] = 0.
    cfg.batch_size = bs
    cfg.N = N
    cfg.hyper_params = hyper
    return cfg
    
