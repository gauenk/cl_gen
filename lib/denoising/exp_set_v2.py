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

def test_get_experiment_set_v2():
    # test has passed!
    a = get_experiment_set_v2(rerun=True)
    b = get_experiment_set_v2(rerun=False)
    return deep_eq(a,b)

def get_experiment_set_v2(rerun=False):
    version = 'v2'
    if check_exp_cache(version) and rerun is False:
        return load_exp_cache(version)

    # load default cfgs
    args = get_args()
    cfg = get_cfg(args)

    # determine experiment grid
    # batch_grid = [1536,768,384,256]
    # batch_grid = [516,258,129,86]
    batch_grid = [2*516,2*258,2*129,2*86]
    N_grid = [2,4,8,12]
    noise_params_grid = [{'stddev':50},{'stddev':100},{'stddev':10}]
    agg_enc_fxn_grid = ['id','mean']
    hyper_parameters_grid = [{'h':0.},{'h':1.}]
    epochs_grid = [500,1000,1500]

    # load the grid
    exps = []
    for noise_params in noise_params_grid:
        for agg_enc_fxn in agg_enc_fxn_grid:
            for hyper in hyper_parameters_grid:
                for bs,N in zip(batch_grid,N_grid):
                    for epochs in epochs_grid:
                        inputs = [cfg,noise_params,bs,N,agg_enc_fxn,hyper]
                        inputs += [epochs]
                        exp = _exp_from_cfg(*inputs)
                        exps.append(exp)

    # save grid to cache
    save_exp_cache(exps,version)
    return exps        

def _exp_from_cfg(_cfg,noise_params,bs,N,agg_enc_fxn,hyper,epochs):
    cfg = copy.deepcopy(_cfg)
    cfg.exp_name = str(uuid.uuid4())
    dsname = cfg.dataset.name.lower()
    cfg.model_path = Path(f"{settings.ROOT_PATH}/output/denoise/{dsname}/{cfg.exp_name}")
    cfg.optim_path = Path(f"{settings.ROOT_PATH}/output/denoise/{dsname}/{cfg.exp_name}/optim/")
    cfg.use_apex = True
    cfg.noise_type = 'g'
    cfg.noise_params['g'] = noise_params
    cfg.noise_params['g']['mean'] = 0.
    cfg.batch_size = bs
    cfg.N = N
    cfg.agg_enc_fxn = agg_enc_fxn
    cfg.hyper_params = hyper
    cfg.epochs = epochs
    return cfg
    
