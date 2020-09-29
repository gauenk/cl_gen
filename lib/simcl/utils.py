# python imports
from easydict import EasyDict as edict

def load_hyperparameters(cfg):
    hyperparams = edict()
    hyperparams.g = 0
    hyperparams.h = cfg.hyper_params['h']
    hyperparams.x = 0
    hyperparams.temperature = 0.1
    return hyperparams

def load_cfg_with_yaml(exp_yml_fn):
    """
    Take an experiment's configuration file (yml)
    and run the corresponding python command.

    Yes, I could just load in the yml rather than read in args...
    but the args are working so this is my mechanism to launch many procs.
    """
    exp_cfg = read_yaml(exp_yml_fn)
    
    cfg.exp_name = args.name
    cfg.epochs = args.epochs
    cfg.load_name = args.load_name
    cfg.epoch_num = args.epoch_num
    cfg.mode = args.mode
    cfg.N = args.N
    cfg.dataset.name = args.dataset
    cfg.batch_size = args.batch_size
    cfg.world_size = args.world_size
    cfg.init_lr = args.init_lr
    cfg.lr_bs_scale = args.lr_bs_scale
    cfg.hyper_params = args.hyper_params
    cfg.noise_type = args.noise_type
    cfg.noise_params = args.noise_params
    cfg.optim_type = args.optim_type
    cfg.optim_params = args.optim_params
    cfg.sched_type = args.sched_type
    cfg.sched_params = args.sched_params
    cfg.num_workers = args.num_workers
    cfg.enc_size = args.enc_size
    cfg.proj_size = args.proj_size
    cfg.freeze_models.encoder = args.freeze_enc
    cfg.freeze_models.projector = args.freeze_proj

