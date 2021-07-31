
# -- python imports --
import copy
from collections import OrderedDict
from easydict import EasyDict as edict

# -- [local] project imports --
from pyutils import apply_mesh_filters,create_named_meshgrid
from datasets.transforms import get_noise_config

def create_mesh():

    # -- alignment methods  --
    align_fxn = ["nn-ransac","nn-flownet_v2","c-frame_v_frame","c-frame_v_mean","c-bootstrap"]

    # -- noise levels --
    noise_type = ['none','g-10p0','g-25p0','g-50p0','g-75p0','g-100p0']
    noise_type += ['qis-40p0-15p0','qis-30p0-15p0','qis-20p0-15p0','qis-4p0-15p0']
    noise_type += ['p-40p0','p-30p0','p-20p0','p-10p0']

    # -- number of frames --
    nframes = [3,5,10,25]

    # -- datasets --
    datasets = ['voc-gm','voc-lm','kitti-default']

    # -- number of blocks --
    nblocks  = [5,7,9,25]
    
    # -- patchsize --
    patchsize = [3,9,32]
    
    # -- npatches --
    npatches = [3]

    # -- ppf --
    ppf = [3]

    # -- random seed --
    random_seed = [123,234,345,456,567]

    # -- batch size --
    batch_size = [10]

    # -- create a list of arrays to mesh --
    lists = [align_fxn,noise_type,nframes,datasets,nblocks,patchsize,npatches,ppf,random_seed,batch_size]
    order = ['align_fxn','noise_type','nframes','datasets','nblocks','patchsize','npatches',
             'ppf','random_seed','batch_size']
    named_list = dict(zip(order,lists))
    
    # -- create mesh --
    named_mesh = create_named_meshgrid(lists,order)

    # -- create filter for nn methods
    non_nn_params = ['nblocks','patchsize','npatches']
    named_mesh = remove_extra_grid_for_nn_align_fxn(named_mesh,named_list,non_nn_params)

    # -- mesh filters --
    filters = [{'nframes-nblocks':[[5,7],[7,7],[7,9],[9,9],[15,15],[15,13],[25,25]]}]
    named_mesh = apply_mesh_filters(named_mesh,filters,'keep')

    return named_mesh,order

def remove_extra_grid_for_nn_align_fxn(named_mesh,non_nn_params,named_list):

    # -- 0.) identify nn align fxns --
    nn_fxn = []
    for align_fxn in named_list['align_fxn']:
        if align_fxn.split("-")[0] == "nn":
            nn_fxn.append(align_fxn)

    # -- 1.) list all but one from extra fields to be removed --
    filters = []
    for name in non_nn_params:
        pairs = []
        uniques = named_list[name]
        for unique in uniques[1:]:
            for fxn in nn_fxn:
                pairs.append([fxn,unique])
        filters.append({"align_fxn-" + name: pairs})

    # -- 2.) remove any non nn fields from each pair --
    named_mesh = apply_mesh_filters(named_mesh,filters,'remove')

    return named_mesh

def config_setup(base_cfg,exp):

    # -- create a copy --
    cfg = copy.deepcopy(base_cfg)

    # -- random seed --
    cfg.random_seed = exp.random_seed

    # -- alignment function --
    cfg.align_fxn = exp.align_fxn

    # -- batchsize --
    cfg.batch_size = int(exp.batch_size)

    # -- set patchsize -- 
    cfg.patchsize = int(exp.patchsize)

    # -- num of patches --
    cfg.npatches = int(exp.npatches)
    
    # -- set frames -- 
    cfg.nframes = int(exp.nframes)
    cfg.N = cfg.nframes

    # -- set number of blocks (old: neighborhood size) -- 
    cfg.nblocks = int(exp.nblocks)
    cfg.nh_size = cfg.nblocks # old name

    # -- set noise params --
    nconfig = get_noise_config(cfg,exp.noise_type)
    cfg.noise_type = nconfig.ntype
    cfg.noise_params = nconfig
    
    # -- dynamics function --
    cfg.frame_size = 128
    cfg.dynamic.ppf = exp.ppf
    cfg.dynamic.bool = True
    cfg.dynamic.random_eraser = False
    cfg.dynamic.frame_size = cfg.frame_size
    cfg.dynamic.total_pixels = cfg.dynamic.ppf*(cfg.nframes-1)
    cfg.dynamic.frames = exp.nframes

    return cfg
