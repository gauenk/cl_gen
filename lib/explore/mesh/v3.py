
# -- python imports --
import copy
from collections import OrderedDict
from easydict import EasyDict as edict

# -- project imports --
from pyutils import create_meshgrid,apply_mesh_filters
from datasets.transforms import get_noise_config

def create_mesh():

    # -- create patchsize grid --
    # patchsize = [13,5]
    patchsize = [64]

    # -- create noise level grid --
    # noise_types = ['pn-4p0-0p0','g-75p0','g-50p0','g-25p0']
    # noise_types = ['pn-4p0-0p0','g-75p0','g-25p0']
    noise_types = ['g-75p0','g-50p0','g-25p0']

    # -- create frame number grid --
    #frames = np.arange(3,9+1,2)
    # nframes = [3,5,7]
    nframes = [7,5,3]

    # -- create number of local regions grid --
    nblocks = [7,9] #np.arange(3,9+1,2)
    
    # -- number of patches from each image --
    npatches = [6]
    
    # -- dynamics grid --
    ppf = [0] #np.arange(3,9+1,2)

    # -- block search grid --
    bss_str = ['0m_5f_200t_d','0m_3f_200t_d'] # mode 0, # for each Frame, # Total, difficult

    # -- image content filter --
    image_decomp_full = ['000'] # emphasis on edges v texture v smooth
    image_decomp_patches = ['000'] # emphasis on edges v texture v smooth
    idf = image_decomp_full
    idp = image_decomp_patches

    # -- batch size --
    batch_size = [10]

    # -- random seed --
    random_seed = [123]

    # -- create a list of arrays to mesh --
    lists = [patchsize,ppf,bss_str,idf,idp,batch_size,
             random_seed,noise_types,nframes,nblocks,npatches]
    order = ['patchsize','ppf','bss_str','idf','idp','batch_size',
             'random_seed','noise_type','nframes','nblocks','npatches']

    # -- create mesh --
    mesh = create_meshgrid(lists)
    
    # -- name each element --
    named_mesh = []
    for elem in mesh:
        named_elem = edict(OrderedDict(dict(zip(order,elem))))
        named_mesh.append(named_elem)

    # -- keep only pairs lists --
    filters = [{'nframes-bss_str':[[3,'0m_5f_200t_d'],[5,'0m_5f_200t_d'],[7,'0m_3f_200t_d']]}]
    named_mesh = apply_mesh_filters(named_mesh,filters)

    return named_mesh,order    

def config_setup(base_cfg,exp):

    # -- create a copy --
    cfg = copy.deepcopy(base_cfg)

    # -- random seed --
    cfg.random_seed = exp.random_seed

    # -- bss --
    cfg.bss_str = exp.bss_str

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
    cfg.frame_size = 196
    cfg.dynamic.ppf = exp.ppf
    cfg.dynamic.bool = True
    cfg.dynamic.random_eraser = False
    cfg.dynamic.frame_size = cfg.frame_size
    cfg.dynamic.total_pixels = cfg.dynamic.ppf*(cfg.nframes-1)
    cfg.dynamic.frames = exp.nframes

    return cfg


"""
INPUTS:

search_grid
- complete or partial
- if partial: how is it partial?
nframes
nblocks
ppf
patchsize
image_content
- full image % decomp into edges v.s. texture v.s. smooth
- patches % decomp into edges v.s. texture v.s. smooth
noise_level & noise_type
batch_size
number_of_patches per image

OUTPUTS:
- plot of frame motion as a function of remaining frame configs
- percent of time the optimum changes as a function of remaining frame configs
- the loss landscape of an INDIVUAL frame motion as a function of remaining
- the loss landscape of an PAIR of frames motion as a function of remaining
- % time individual frames agree with optima
- % time aggregate score agrees with optima
- comparing individual frame and aggregate scores
- score_function
    model_type
    - burst_model
    - frame_model
    model_train
    - subset choices (cog only)
    - num iters


FEATURES:
- allow us to grab the same patch of the same images upon request

"""

