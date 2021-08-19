from patch_search import get_score_function


def bootstrapping(cfg,patches):
    score_fxn = get_score_function("bootstrapping_mod2")
    return run_scores_fxn(cfg,patches,score_fxn)

def l2(cfg,patches):
    score_fxn = get_score_function("ave")
    return run_scores_fxn(cfg,patches,score_fxn)

def run_scores_fxn(cfg,patches,score_fxn):
    nimages = patches.shape[0]
    patches = rearrange(patches,'b p t a c h w -> 1 (b p) a t c h w')
    cfg = edict({'gpuid':gpuid})
    scores,scores_t = score_fxn(cfg,patches)
    scores = rearrange(scores,'1 (b p) a -> b p a',b=nimages)
    return scores

    
