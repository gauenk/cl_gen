"""

This code will match a dictionary ("exp_config")
with a uuid

"""

from ._utils import compare_config

def get_uuid_from_config(data,exp_config):

    """

    Given a config (or python dictionary), we return the associated uuid

    """

    if data is None:
        raise ValueError("[uuid_cache/_convert.py get_uuid_from_config]: Data is None.")
    for uuid,config in zip(data.uuid,data.config):
        match = compare_config(config,exp_config)
        if match: return uuid
    return -1 # no match

def get_config_from_uuid(data,exp_uuid):

    """

    Given a uuid, we return the associated config (or python dictionary).

    """

    if data is None:
        raise ValueError("[uuid_cache/_convert.py get_config_from_uuid]: Data is None.")
    for uuid,config in zip(data.uuid,data.config):
        if uuid == exp_uuid: return config
    return -1 # no match


