
# -- python imports --
import pandas as pd

from ._debug import VERBOSE
from ._write import *
from ._convert import *
from ._utils import *

__all__ = [append_new_pair,set_new_field_default,set_new_field_data]

def append_new_pair(data,uuid_file,new_pair):
    existing_uuid = get_uuid_from_config(data,new_pair.config)
    existing_config = get_config_from_uuid(data,new_pair.uuid)
    if existing_uuid == -1 and existing_config == -1:
        data.uuid.append(new_pair.uuid)
        data.config.append(new_pair.config)
        write_uuid_file(uuid_file,data)
    else:
        if VERBOSE: print("Not appending data to file since data already exists.")
        if existing_uuid != -1 and VERBOSE:
            print(f"UUID already exists: [{new_pair.uuid}]")
        if existing_config == -1 and VERBOSE:
            print(f"Config already exists")
            print_config(new_pair.config)

def set_new_field_default(data,new_field,default):
    configs = pd.DataFrame(data.config)
    keys = list(configs.columns)
    if new_field in keys:
        if VERBOSE: print(f"Not appending new field [{new_field}]. Field already exists.")
        return -1
    configs[new_field] = default
    data.config = configs.to_dict()
    return 1

def set_new_field_data(data,new_data,new_field):
    for uuid,new_results in zip(new_data.uuid,new_data.results):
        index = np.where(data.uuid == uuid)[0]
        data.config[index][new_field] = new_results
