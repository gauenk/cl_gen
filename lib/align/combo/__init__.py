
# -- python imports --

# -- pytorch imports --

# -- project imports --
import align.combo.lpas
import align.combo.nlpas
import align.combo.eval_scores
from align._utils import construct_return_dict

def combo_compute_alignment(cfg,images):

    aligned = images
    flow = []

    T,B,C,H,W = images.shape

    ref_frame = T//2
    nblocks = cfg.nblocks
    
    lpas.lpas_search(images,ref_frame,nblocks,motion=None,
                     method="simple",noise_info=None)
    
    options = {"aligned":aligned,"flow":flow}
    results = construct_return_dict(fields,options)

    return results

def assert_combo_fields(cfg,afxn):
    """
    Ensure all fields required for NN alignment exist.

    Listing out specific fields reduces black-box dependence on cfg.

    """
    if afxn == "lpas":
        return lpas.assert_cfg_fields(cfg)
    elif afxn == "nlaps":
        return nlpas.assert_cfg_fields(cfg)
    else:
        raise ValueError(f"Uknown combinatorial alignment function [{afxn}]")


