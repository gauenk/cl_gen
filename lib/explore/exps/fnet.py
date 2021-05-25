
# -- python --
from einops import rearrange

# -- pytorch --
import torch

# -- project --
from layers.unet import UNet_n2n,UNet_small
from patch_search.fnet import run_fnet as run_fnet_ps
from patch_search import get_score_function

def run_fnet(cfg,clean,noisy,directions,results):

    # -- explicit shape --
    E,T,B,P,C,PS,PS = clean.shape
    clean = clean[0,:,0,0]
    noisy = noisy[0,:,0,0]
    # clean = rearrange(clean[:,:,0],'t b c ps1 ps2 -> b t c ps1 ps2')
    # noisy = rearrange(noisy[:,:,0],'t b c ps1 ps2 -> b t c ps1 ps2')

    # -- create grid of score functions --
    score_fxn_names = ['ave','lgsubset_v_ref']

    # -- fnet over score fxns --
    for score_fxn_name in score_fxn_names:
        run_fnet_score(cfg,score_fxn_name,clean,noisy,directions,results)

def run_fnet_score(cfg,score_fxn_name,clean,noisy,directions,results):
    score_fxn = get_score_function(score_fxn_name)
    score,scores_t = run_fnet_ps(cfg,noisy,score_fxn)
    T = len(scores_t)
    results['fnet_{score_fxn_name}_score'] = score
    for t in range(T): results[f'fnet_{score_fxn_name}_score_{t}'] = scores_t[t]

