
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
    cfg.nframes = 10
    cfg.frame_size = 16

    # -- data config --
    cfg.dataset = edict()
    cfg.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.dataset.name = "voc"
    cfg.dataset.dict_loader = True
    cfg.dataset.num_workers = 2
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
    cfg.score_fxn_name = "bootstrapping_mod2"
    # cfg.score_fxn_name = "bootstrapping_mod3"
    # cfg.score_fxn_name = "bootstrapping_mod4"

    return cfg

def get_exp_cfgs(name):
    # ignore name for now.

    # -- create patchsize grid --
    # patchsize = [3,5]
    # patchsize = [3,5,7,15]
    patchsize = [11]
    ps_ticks = patchsize
    ps_tickmarks = ps_ticks
    ps_tickmarks_str = ["%d" % x for x in ps_tickmarks]

    # -- create noise level grid --
    # noise_types = ['pn-4p0-0p0','g-75p0','g-50p0','g-25p0']
    # noise_types = ['pn-4p0-0p0','g-75p0','g-25p0']
    # noise_types = ['g-150p','g-100p','g-75p0','g-50p0','g-25p0','g-5p0']
    # noise_types = ['g-100p0','g-5p0']
    noise_types = ['g-5p0','g-10p0','g-15p0','g-20p0']
    # noise_types = ['g-75p0','g-25p0','g-5p0']
    # std_ticks = [5.,25.,50.,75.,100.,150.]
    std_ticks = [5.,25.,75.,150.]
    std_tickmarks = std_ticks
    std_tickmarks_str = ["%d" % x for x in std_tickmarks]

    alpha_ticks = [-1,-1,-1,-1,-1]
    alpha_tickmarks = alpha_ticks
    alpha_tickmarks_str = ["%d" % x for x in alpha_tickmarks]

    # -- create frame number grid --
    nframes = [10,3] # [31]
    nframes_ticks = nframes
    nframes_tickmarks = nframes_ticks
    nframes_tickmarks_str = ["%d" % x for x in nframes_tickmarks]

    # -- bootstrapping name --
    bsname = ['bootstrapping_mod2']

    # -- create number of local regions grid --
    nblocks = [3]
    
    # -- dynamics grid --
    ppf = [0]

    # -- batch size --
    batch_size = [1]

    # -- random seed --
    random_seed = [234,345,456]

    # -- create a list of arrays to mesh --
    lists = [patchsize,noise_types,nframes,nblocks,
             bsname,ppf,batch_size,random_seed]
    order = ['patchsize','noise_type','nframes','nblocks',
             'bsname','ppf','batch_size','random_seed']
    named_params = edict({o:l for o,l in zip(order,lists)})


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

    # -- format grids --
    logs = {'nframes':False,'patchsize':True,'std':False,'alpha':False}
    ticks = edict({'nframes':nframes_ticks,'patchsize':ps_ticks,
                   'std':std_ticks,'alpha':alpha_ticks})
    tickmarks = edict({'nframes':nframes_tickmarks,
                       'patchsize':ps_tickmarks,
                       'std':std_tickmarks,
                       'alpha':alpha_tickmarks,})
    tickmarks_str = edict({'nframes':nframes_tickmarks_str,
                           'patchsize':ps_tickmarks_str,
                           'std':std_tickmarks_str,
                           'alpha':alpha_tickmarks_str,})
    egrids = edict({'ticks':ticks,'tickmarks':tickmarks,
                   'tickmarks_str':tickmarks_str,'logs':logs,
                   'grids':named_params})

    return named_mesh,order,egrids

def setup_exp_cfg(base_cfg,exp):

    # -- create a copy --
    cfg = copy.deepcopy(base_cfg)

    # -- random seed --
    cfg.random_seed = exp.random_seed

    # -- number of frames --
    cfg.nframes = int(exp.nframes)

    # -- bootstrapping name --
    cfg.score_fxn_name = exp.bsname

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
    cfg.dynamic_info.ppf = exp.ppf
    cfg.dynamic_info.total_pixels = cfg.dynamic_info.ppf*(cfg.nframes-1)
    cfg.dynamic_info.nframes = exp.nframes

    return cfg

