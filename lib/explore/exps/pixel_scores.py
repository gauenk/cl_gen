
# -- python --
from einops import rearrange

# -- pytorch --

# -- project --
from patch_search import get_score_function

def run_pixel_scores(cfg,clean,noisy,directions,results):
    P,B,E,T,C,PS,PS = clean.shape # input shape
    # clean = rearrange(clean,'t e b p c h w -> p b e t c h w')
    # noisy = rearrange(noisy,'t e b p c h w -> p b e t c h w')
    """
    P = different patches from same image ("R" in patch_match/pixel/score.py)
    B = different images
    E = differnet block regions around centered patch (batch of grid)
    T = burst of frames along a batch dimension
    """
    # score_fxn_names = ["ave","lgsubset","lgsubset_v_indices","lgsubset_v_ref"]
    # score_fxn_names = ["ave","lgsubset_v_ref","jackknife"]
    # score_fxn_names = ["ave","jackknife"]
    # score_fxn_names = ["ave","sim_trm"]
    score_fxn_names = ["ave","ransac"]
    # score_fxn_names = ["ransac"]
    for score_fxn_name in score_fxn_names:
        score_fxn = get_score_function(score_fxn_name)
        scores,scores_t = score_fxn(cfg,noisy)
        scores,scores_t = scores.cpu(),scores_t.cpu()
        fieldname = f'pixel_{score_fxn_name}'
        results[fieldname] = {'scores':scores,'scores_t':scores_t}

    
