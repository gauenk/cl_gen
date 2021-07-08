
import pprint
from pathlib import Path
from ._debug import VERBOSE

def compare_config(existing_config,proposed_config):
    for key,value in existing_config.items():
        if proposed_config[key] != value: return False
    return True

def print_config(config,indent=1):
    pp = pprint.PrettyPrinter(depth=5,indent=indent)
    pp.pprint(config)



