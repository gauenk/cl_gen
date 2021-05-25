
import json
import uuid as uuid_gen
from easydict import EasyDict as edict
from .config import cfg
from .utils import print_formatted_exp_info

verbose = False

def init_uuid_file(cfg):
    if verbose: print(f"Init [{cfg.uuid_file}]")
    if cfg.uuid_file.exists(): return None    
    data = edict({'uuid':[],'config':[]})
    write_uuid_file(cfg,data)

def get_uuid_from_config(exp_config):
    data = read_uuid_file(cfg)
    if data is None:
        init_uuid_file(cfg)
        return -1
    for uuid,config in zip(data.uuid,data.config):
        match = compare_config(config,exp_config)
        if match: return uuid
    return -1 # no match

def get_config_from_uuid(exp_uuid):
    data = read_uuid_file(cfg)
    if data is None:
        init_uuid_file(cfg)
        return -1
    for uuid,config in zip(data.uuid,data.config):
        if uuid == exp_uuid: return config
    return -1 # no match

def compare_config(existing_config,proposed_config):
    for key,value in existing_config.items():
        if proposed_config[key] != value: return False
    return True

def get_uuid(exp_config):
    uuid = get_uuid_from_config(exp_config)    
    if uuid == -1:
        if verbose: print("Creating a new UUID and adding to cache file.")
        uuid = str(uuid_gen.uuid4())
        new_pair = edict({'uuid':uuid,'config':exp_config})
        append_new_pair(cfg,new_pair)
        return uuid
    else:
        if verbose: print(f"Exp Config Already has a UUID {uuid}")
        return uuid

def read_uuid_file(cfg):
    if verbose: print(f"Reading: [{cfg.uuid_file}]")
    if not cfg.uuid_file.exists(): return None
    with open(cfg.uuid_file,'r') as f:
        data = edict(json.load(f))
    return data

def write_uuid_file(cfg,data):
    if verbose: print(f"Writing: [{cfg.uuid_file}]")
    if not cfg.uuid_file.parents[0].exists():
        cfg.uuid_file.parents[0].mkdir(parents=True)
    with open(cfg.uuid_file,'w') as f:
        json.dump(data,f)

def append_new_field(cfg,version,field,default,new_data):

    # -- open stored data --
    default_version = cfg.version
    data = read_uuid_file(cfg)

    # -- check if key exists & set default --
    configs = pd.DataFrame(data.config)
    keys = list(configs.columns)
    if field in keys:
        if verbose: print(f"Not appending new field [{field}]. Field already exists.")
        return None
    configs[field] = default
    data.config = configs.to_dict()

    # -- apply non-default values to data --
    for uuid,new_results in zip(new_data.uuid,new_data.results):
        index = np.where(data.uuid == uuid)[0]
        data.config[index][field] = new_results

    # -- write updated uuid file --
    cfg.version = version
    write_uuid_file(cfg,data)
    cfg.version = default_version
    if verbose: print(f"Wrote new uuid cache with new field. Version [{version}]")

    return 

def append_new_pair(cfg,new_pair):
    data = read_uuid_file(cfg)
    existing_uuid = get_uuid_from_config(new_pair.config)
    existing_config = get_config_from_uuid(new_pair.uuid)
    if existing_uuid == -1 and existing_config == -1:
        data.uuid.append(new_pair.uuid)
        data.config.append(new_pair.config)
        write_uuid_file(cfg,data)
    else:
        if verbose: print("Not appending data to file since data already exists.")
        if existing_uuid != -1 and verbose:
            print(f"UUID already exists: [{new_pair.uuid}]")
        if existing_config == -1 and verbose:
            print(f"Config already exists")
            print_formatted_exp_info(new_pair.config)


def get_config_from_uuid_list(uuids):
    configs = []
    for uuid in uuids:
        configs.append(get_config_from_uuid(uuid))
    return configs
