
# -- python --

# -- pytorch --

# -- project --
from layers.unet import UNet_n2n,UNet_small
from patch_search.cog import score_cog

def run_cog(cfg,clean,noisy,directions,results):

    image_volume = noisy
    backbone = UNet_small(3).to(cfg.device)
    nn_params = {'lr':1e-3,'init_params':None}
    train_steps = 1000
    score = score_cog(cfg,image_volume,backbone,nn_params,train_steps)
    results['cog'] = score

    return 
    
