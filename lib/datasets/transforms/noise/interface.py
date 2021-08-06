import copy
from easydict import EasyDict as edict

def get_keys_noise_level_grid(cfg):
    noise_configs = create_noise_level_grid(cfg)
    keys = []
    for noise_config in noise_configs:
        keys.append(noise_config['name'])
    return keys

def set_noise(cfg,ns):
    cfg.noise_params.ntype = ns.ntype
    if ns.ntype == "g":
        cfg.noise_params[ns.ntype]['stddev'] = ns.stddev
    elif ns.ntype == "hg":
        cfg.noise_params[ns.ntype]['read'] = ns.read
        cfg.noise_params[ns.ntype]['shot'] = ns.shot
    elif ns.ntype == "pn":
        cfg.noise_params[ns.ntype]['alpha'] = ns.alpha
        cfg.noise_params[ns.ntype]['std'] = ns.std
    elif ns.ntype == "qis":
        cfg.noise_params[ns.ntype]['alpha'] = ns.alpha
        cfg.noise_params[ns.ntype]['readout'] = ns.readout
        cfg.noise_params[ns.ntype]['nbits'] = ns.nbits
    elif ns.ntype == "none":
        return ns.name
    else:
        raise ValueError(f"Uknown noise type [{ns.ntype}]")
    return ns.name

def get_noise_config(cfg,name):

    # -- 1.) create default config from name --
    if name.split("-")[0] == "g": # gaussian
        wrapper = edict()
        config = get_gaussian_config_from_name(cfg,name)
        wrapper[config.ntype] = config
        wrapper.ntype = config.ntype
        return wrapper
    elif name.split("-")[0] == "pn":
        wrapper = edict()
        config = get_poisson_config_from_name(cfg,name)
        wrapper[config.ntype] = config
        wrapper.ntype = config.ntype
        return wrapper

    # -- 2.) search for noise in fixed params --
    nconfigs = create_noise_level_grid(cfg)
    for config in nconfigs:
        if name == config.name:
            wrapper = edict()
            wrapper[config.ntype] = config
            wrapper.ntype = config.ntype
            return wrapper
    
    raise ValueError(f"Uknown noise type [{name}]")

def create_noise_level_grid(cfg):
    noise_configs = []

    # -- no noise --
    noise_type = 'none'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['name'] = "clean"
    ns = edict(ns)
    noise_configs.append(ns)

    # -- no noise --
    noise_type = 'none'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['name'] = "none"
    ns = edict(ns)
    noise_configs.append(ns)

    # -- gaussian noise --
    noise_type = 'g'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['stddev'] = 125.
    ns['name'] = f"g-{ns['stddev']}".replace(".","p")
    ns = edict(ns)
    noise_configs.append(ns)

    # -- gaussian noise --
    noise_type = 'g'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['stddev'] = 100.
    ns['name'] = f"g-{ns['stddev']}".replace(".","p")
    ns = edict(ns)
    noise_configs.append(ns)

    # -- gaussian noise --
    noise_type = 'g'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['stddev'] = 75.
    ns['name'] = f"g-{ns['stddev']}".replace(".","p")
    ns = edict(ns)
    noise_configs.append(ns)
    
    # -- gaussian noise --
    noise_type = 'g'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['stddev'] = 25.
    ns['name'] = f"g-{ns['stddev']}".replace(".","p")
    ns = edict(ns)
    noise_configs.append(ns)

    # -- gaussian noise --
    noise_type = 'g'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['stddev'] = 50.
    ns['name'] = f"g-{ns['stddev']}".replace(".","p")
    ns = edict(ns)
    noise_configs.append(ns)

    # -- poisson noise --
    noise_type = 'pn'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['alpha'] = 4.
    ns['std'] = 0.
    ns['name'] = f"pn-{ns['alpha']}-{ns['std']}"
    ns['name'] = ns['name'].replace(".","p")
    ns = edict(ns)
    noise_configs.append(ns)

    return noise_configs

def get_gaussian_config_from_name(cfg,name):
    noise_type,noise_str = name.split("-")
    noise_level = float(noise_str.replace('p','.'))
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['stddev'] = noise_level # legacy
    ns['std'] = noise_level
    ns['name'] = f"g-{ns['std']}".replace(".","p")
    ns = edict(ns)
    return ns

def get_poisson_config_from_name(cfg,name):
    noise_type,rate_str,std_str = name.split("-")
    rate_level = float(rate_str.replace('p','.'))
    std_level = float(std_str.replace('p','.'))
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['alpha'] = rate_level
    ns['std'] = std_level
    ns['name'] = name
    ns = edict(ns)
    return ns

