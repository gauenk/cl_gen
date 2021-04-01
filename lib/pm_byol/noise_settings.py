import copy
from easydict import EasyDict as edict

def get_keys_noise_level_grid(cfg):
    noise_settings = create_noise_level_grid(cfg)
    keys = []
    for noise_setting in noise_settings:
        keys.append(noise_setting['name'])
    return keys

def create_noise_level_grid(cfg):
    noise_settings = []

    # -- gaussian noise --
    noise_type = 'none'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['name'] = "clean"
    ns = edict(ns)
    noise_settings.append(ns)

    # -- gaussian noise --
    noise_type = 'g'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['stddev'] = 75.
    ns['name'] = f"g-{ns['stddev']}".replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)
    
    # -- poisson noise --
    noise_type = 'pn'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['alpha'] = 4.
    ns['std'] = 0.
    ns['name'] = f"pn-{ns['alpha']}-{ns['std']}"
    ns['name'] = ns['name'].replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)

    return noise_settings
