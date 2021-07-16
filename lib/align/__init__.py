
# -- python imports --

# -- pytorch imports --

# -- project imports --
import align.nn
import align.combo
from align._utils import assert_cfg_fields,compute_epe,compute_aligned_psnr

def compute_alignment(cfg,align_fxn,images):
    """

    align_fxn anatomy is "{alignment_function_type}-{alignment_function_name}"

    The {alignment_function_type} is either:
    1.) "nn" for neural network
    2.) "c" for combinatorial

    """
    assert_cfg_fields(cfg)

    atype,afxn = align_fxn.split("-")
    if atype == "nn":
        nn.assert_nn_fields(cfg,afxn)
        return nn.nn_compute_alignment(cfg,images)
    elif atype == "c":
        combo.assert_combo_fields(cfg,afxn)
        return combo.combo_compute_alignment(cfg,images)
    else:
        raise ValueError("Uknown alignment function type [{atype}]")

