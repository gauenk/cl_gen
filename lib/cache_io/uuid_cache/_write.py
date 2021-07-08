
# -- python imports --
import json
from pathlib import Path
from ._debug import VERBOSE

def write_uuid_file(uuid_file,data):
    if VERBOSE: print(f"Writing: [{cfg.uuid_file}]")
    if not uuid_file.parents[0].exists():
        uuid_file.parents[0].mkdir(parents=True)
    with open(uuid_file,'w') as f:
        json.dump(data,f)


