"""

Results from Experiment Configurations are Cached in UUID Directories
.../output/lpas/explore/uuid_0/results.pkl

Directories are UUID Values
.../output/lpas/explore/uuid_0
.../output/lpas/explore/uuid_1
.../output/lpas/explore/uuid_1
...

UUID Values map to Experiment Configurations
.../output/lpas/explore/uuid_database.json
   config <-> uuid

Experiment Configurations Explain how an experiment was run

score_fxn = 'asdf'
patchsize = 1
nbatches = 10
image_id = 0,1,2
random_seed = 123
...


Inspect Using a Tool
lsc: quickly "ls" a uuid to get experiment config info

"""

import shutil,os
from pathlib import Path

# -> io using only the *experiment config*
# -> *uuid* is an INTERNAL tool used for io
from .save import save_exp
from .load import load_exp
from .uuid_cache import read_uuid_file,get_uuid
from .tensor_cache import append_tensor_cache
from .config import cfg

def clear_cache():
    print("Clearing Cache.")
    uuid_file = cfg.uuid_file
    if not uuid_file.exists(): return

    # -- remove all experiment results --
    data = read_uuid_file(cfg)
    for uuid in data.uuid:
        path = cfg.root / Path(uuid)
        if not path.exists(): continue
        shutil.rmtree(path)
        assert not path.exists(),f"exp cache [{path}] should be removed."

    # -- remove uuid cache --
    if uuid_file.exists(): os.remove(uuid_file)
    assert not uuid_file.exists(),f"uuid file [{uuid_file}] should be removed."

