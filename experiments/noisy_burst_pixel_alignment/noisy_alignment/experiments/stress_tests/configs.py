
# -- python imports --
import copy
from collections import OrderedDict
from easydict import EasyDict as edict

# -- project imports --
import settings
from pyutils import create_meshgrid,apply_mesh_filters
from datasets.transforms import get_noise_config

def get_cfg_defaults():
    cfg = edict()

    # -- frame info --
    cfg.nframes = 15
    cfg.frame_size = 32

    # -- data config --
    cfg.dataset = edict()
    cfg.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.dataset.name = "voc"
    cfg.dataset.dict_loader = True
    cfg.dataset.num_workers = 2
    cfg.batch_size = 1
    cfg.drop_last = {'tr':True,'val':True,'te':True}
    cfg.noise_params = edict({'pn':{'alpha':10.,'std':0},
                              'g':{'std':25.},'ntype':'pn'})
    cfg.dynamic_info = edict()
    cfg.dynamic_info.mode = 'jitter'
    cfg.dynamic_info.frame_size = cfg.frame_size
    cfg.dynamic_info.nframes = cfg.nframes
    cfg.dynamic_info.ppf = 0
    cfg.dynamic_info.textured = True
    cfg.random_seed = 234

    # -- combo config --
    cfg.nblocks = 3
    cfg.patchsize = 5
    # cfg.score_fxn_name = "bootstrapping"
    cfg.score_fxn_name = "bootstrapping_mod3"
    
    return cfg

def get_exp_cfgs(name):

    # -- create patchsize grid --
    patchsize = [3,5]

    # -- create noise level grid --
    # noise_types = ['pn-4p0-0p0','g-75p0','g-50p0','g-25p0']
    # noise_types = ['pn-4p0-0p0','g-75p0','g-25p0']
    noise_types = ['g-75p0','g-50p0','g-25p0']

    # -- create frame number grid --
    nframes = [7,5,3]

    # -- create number of local regions grid --
    nblocks = [3]
    
    # -- dynamics grid --
    ppf = [0]

    # -- batch size --
    batch_size = [10]

    # -- random seed --
    random_seed = [123,234]

    # -- create a list of arrays to mesh --
    lists = [patchsize,noise_types,nframes,nblocks,
             ppf,batch_size,random_seed]
    order = ['patchsize','noise_type','nframes','nblocks',
             'ppf','batch_size','random_seed']

    # -- create mesh --
    mesh = create_meshgrid(lists)
    
    # -- name each element --
    named_mesh = []
    for elem in mesh:
        named_elem = edict(OrderedDict(dict(zip(order,elem))))
        named_mesh.append(named_elem)

    # -- keep only pairs lists --
    # filters = [{'nframes-ppf':[[3,3],[5,1]]}]
    # named_mesh = apply_mesh_filters(named_mesh,filters)

    return named_mesh,order    

def setup_exp_cfg():

    # -- create a copy --
    cfg = copy.deepcopy(base_cfg)

    # -- random seed --
    cfg.random_seed = exp.random_seed

    # -- number of frames --
    cfg.nframes = int(exp.nframes)

    # -- batchsize --
    cfg.batch_size = int(exp.batch_size)

    # -- combinatoric search info --
    cfg.nblocks = int(exp.nblocks)
    cfg.patchsize = int(exp.patchsize)

    # -- set noise params --
    nconfig = get_noise_config(cfg,exp.noise_type)
    cfg.noise_type = nconfig.ntype
    cfg.noise_params = nconfig
    
    # -- dynamics function --
    cfg.frame_size = 196
    cfg.dynamic.ppf = exp.ppf
    cfg.dynamic.bool = True
    cfg.dynamic.random_eraser = False
    cfg.dynamic.frame_size = cfg.frame_size
    cfg.dynamic.total_pixels = cfg.dynamic.ppf*(cfg.nframes-1)
    cfg.dynamic.nframes = exp.nframes

    return cfg

