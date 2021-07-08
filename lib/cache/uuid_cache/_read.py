
import json
from pathlib import Path
from easydict import EasyDict as edict

from ._debug import VERBOSE

def read_uuid_file(uuid_file):
    if VERBOSE: print(f"Reading: [{uuid_file}]")
    if not uuid_file.exists(): return None
    with open(uuid_file,'r') as f:
        data = edict(json.load(f))
    return data



