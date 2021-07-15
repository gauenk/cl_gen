# -- python imports --

# -- pytorch imports --

# -- project imports --
import align.nn.nn_fxn

def nn_compute_alignment(cfg,images):
    pass

def assert_nn_fields(cfg):
    """
    Ensure all fields required for NN alignment exist.

    Listing out specific fields reduces black-box dependence on cfg.
    """
    
    if afxn == "raft":
        return nn_fxn.raft.assert_cfg_fields(cfg)
    elif afxn == "flownet_v2":
        return nn_fxn.flownet_v2.assert_cfg_fields(cfg)
    else:
        raise ValueError(f"Uknown neural network alignment function [{afxn}]")

