
#
# Interface
#

def get_score_functions(names):
    scores = edict()
    for name in names:
        scores[name] = get_score_function(name)
    return scores

def get_score_function(name):
    if name == "fast_unet_ave":
        return fast_unet_ave
    elif name == "fast_unet_lgsubset":
        return fast_unet_lgsubset
    elif name == "fast_unet_lgsubset_v_indices":
        return fast_unet_lgsubset_v_indices
    elif name == "fast_unet_lgsubset_v_ref":
        return fast_unet_lgsubset_v_ref
    else:
        raise ValueError(f"Uknown score function [{name}]")

#
# Fast UNet Scores
#

def fast_unet_search(cfg,expanded,search_method):
    R,B,E,T,C,H,W = expanded.shape
    assert (R == 1) and (B == 1), "Must have one patch and one batch item."
    scores_t = torch.zeros(R,B,E,T)
    scores = torch.zeros(R,B,E)
    for e in range(E):
        burst = expanded[0,0,e]
        score,scores_t = run_fast_unet(cfg,burst,search_method)
        scores_t[0,0,e] = scores_t
        scores[0,0,e] = score
    return scores

def fast_unet_ave(cfg,expanded):
    search_method = get_score_function("ave")
    return fast_unet_search(cfg,expanded,search_method)

def fast_unet_lgsubset(cfg,expanded):
    search_method = get_score_function("lgsubset")
    return fast_unet_search(cfg,expanded,search_method)

def fast_unet_lgsubset_v_indices(cfg,expanded):
    search_method = get_score_function("lgsubset_v_indices")
    return fast_unet_search(cfg,expanded,search_method)

def fast_unet_lgsubset_v_ref(cfg,expanded):
    search_method = get_score_function("lgsubset_v_ref")
    return fast_unet_search(cfg,expanded,search_method)
