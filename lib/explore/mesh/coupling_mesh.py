# -- python imports --
from easydict import EasyDict as edict

# -- [local] project imports --
from .utils import create_meshgrid

def create_coupling_mesh():

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
              'lgsubset_v_ref','lgsubset','ave','lgsubset_v_indices',
              'cog_burst_v1','cog_frame_v1']
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
    - % decomp into edges v.s. texture v.s. smooth
    model_type
    - burst_model
    - frame_model
    model_train
    - subset choices (cog only)
    - num iters
    noise_level
    noise_type
    score_function
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

    FEATURES:
    - allow us to grab the same patch of the same images upon request
    
    
    """

    # -- create patchsize grid --
    # psgrid = [13,5]
    psgrid = [16]

    # -- create noise level grid --
    # noise_types = ['pn-4p0-0p0','g-75p0','g-50p0','g-25p0']
    # noise_types = ['pn-4p0-0p0','g-75p0','g-25p0']
    noise_types = ['g-75p0','g-50p0','g-25p0']

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
