
# -- python imports --
from easydict import EasyDict as edict

# -- project imports --
import settings 

def get_cfg_defaults():
    cfg = edict()

    # -- frame info --
    cfg.nframes = 5
    cfg.use_anscombe = True
    cfg.frame_size = 64
    cfg.nepochs = 10
    # cfg.nepochs = 3
    cfg.test_interval = 2
    cfg.save_interval = 5
    cfg.train_log_interval = 50
    cfg.test_log_interval = 10
    cfg.global_step = 0
    cfg.gpuid = 1

    # -- data config --
    cfg.dataset = edict()
    cfg.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.dataset.name = "voc"
    cfg.dataset.dict_loader = True
    cfg.dataset.num_workers = 0
    cfg.batch_size = 1
    cfg.drop_last = {'tr':True,'val':True,'te':True}
    cfg.noise_params = edict({'pn':{'alpha':10.,'std':0},
                              'g':{'std':25.0},'ntype':'pn'})
    cfg.dynamic_info = edict()
    cfg.dynamic_info.mode = 'global'
    cfg.dynamic_info.frame_size = cfg.frame_size
    cfg.dynamic_info.nframes = cfg.nframes
    cfg.dynamic_info.ppf = 0
    cfg.dynamic_info.textured = True

    cfg.random_seed = 0

    # -- combo config --
    cfg.nblocks = 3
    cfg.patchsize = 3
    # cfg.score_fxn_name = "bootstrapping"
    cfg.score_fxn_name = "bootstrapping_mod3"
    return cfg
