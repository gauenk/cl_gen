
from .pixel import get_score_function_pixel
from .fnet import get_score_function_fnet
from .cog import get_score_function_cog

def get_score_functions(names):
    scores = edict()
    for name in names:
        scores[name] = get_score_function(name)
    return scores

def get_score_function(name):
    fxn_pixel = get_score_function_pixel(name)
    if not(fxn_pixel is None): return fxn_pixel
    fxn_fnet = get_score_function_fnet(name)
    if not(fxn_fnet is None): return fxn_fnet
    fxn_cog = get_score_function_cog(name)    
    if not(fxn_cog is None): return fxn_cog
    raise ValueError(f"Uknown score function [{name}]")
