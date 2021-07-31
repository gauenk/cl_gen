
# -- python imports --

# -- pytorch imports --

# -- project imports --
import lpas


def compute_alignment(cfg,align_fxn,images):
    atype = align_fxn.split("-")[0]
    if atype == "nn":
        return nn_compute_alignment(cfg,images)
    elif atype == "c":
        return combinatorial_compute_alignment(cfg,images)
    else:
        raise ValueError("Uknown alignment function type [{atype}]")

def nn_compute_alignment(cfg,images):
    
    pass

def combinatorial_compute_alignment(cfg,images):
    pass

    aligned = images
    flow = []

    T,B,C,H,W = images.shape

    ref_frame = T//2
    nblocks = cfg.nblocks
    
    lpas.lpas_search(images,ref_frame,nblocks,motion=None,method="simple",
                to_align=None,noise_info=None)

    return aligned,flow
