#
# Interface
#

def get_score_functions(names):
    scores = edict()
    for name in names:
        scores[name] = get_score_function(name)
    return scores

def get_score_function(name):
    if name == "cog_frame_v1":
        return cog_frame_v1
    else:
        return None

#
# Consistency of denoiser Graphs Based Losses
# 

def cog_search(cfg,expanded,search_method):
    R,B,E,T,C,H,W = expanded.shape
    assert (R == 1) and (B == 1), "Must have one patch and one batch item."
    scores_t = torch.zeros(R,B,E,T)
    scores = torch.zeros(R,B,E)

    # -- cog params --
    backbone = UNet_small
    nn_params = edict({'lr':1e-3,'init_params':None})
    train_steps = 1000
    for e in range(E):
        burst = expanded[0,0,e]
        score,scores_t = score_cog(cfg,burst,backbone,nn_params,train_steps)
        scores_t[0,0,e] = scores_t
        scores[0,0,e] = score
    return scores

def cog_frame_v1(cfg,expanded):
    return cog_search(cfg,expanded,"v1")

