# -- python imports --
from easydict import EasyDict as edict

# -- [local] project imports --
from .utils import create_meshgrid

def create_eval_mesh():

    # -- create score function grid --
    # scores = ['ave','got','emd','powerset']
    # scores = ['ave','pairwise','refcmp']#,'powerset']
    # scores = ['powerset','ave','extrema','lgsubset','lgsubset_v_indices',
    #           'lgsubset_v_ref','powerset_v_indices']
    # scores = ['lgsubset_v_ref','extrema','lgsubset','ave','lgsubset_v_indices']
    # scores = ['lgsubset_v_ref','lgsubset','ave','lgsubset_v_indices',
    #           'fast_unet_lgsubset','fast_unet_lgsubset_v_indices','fast_unet_ave',
    #           'fast_unet_lgsubset_v_ref']
    scores = ['fast_unet_lgsubset','fast_unet_lgsubset_v_indices','fast_unet_ave',
              'fast_unet_lgsubset_v_ref',
              'lgsubset_v_ref','lgsubset','ave','lgsubset_v_indices',]

    # -- create patchsize grid --
    # psgrid = [13,5]
    psgrid = [16]

    # -- create noise level grid --
    # noise_types = ['pn-4p0-0p0','g-75p0','g-50p0','g-25p0']
    noise_types = ['pn-4p0-0p0','g-75p0','g-25p0']

    # -- create frame number grid --
    #frames = np.arange(3,9+1,2)
    frames = [3,5,7]

    # -- create number of local regions grid --
    blocks = [3,5,7] #np.arange(3,9+1,2)
    
    # -- dynamics grid --
    ppf = [1] #np.arange(3,9+1,2)

    # -- create a list of arrays to mesh --
    lists = [scores,psgrid,noise_types,frames,blocks,ppf]
    order = ['score_function','patchsize','noise_type','nframes','nblocks','ppf']

    # -- create mesh --
    mesh = create_meshgrid(lists)
    
    # -- name each element --
    named_mesh = []
    for elem in mesh:
        named_elem = edict(dict(zip(order,elem)))
        named_mesh.append(named_elem)

    return named_mesh,order
