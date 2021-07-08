
# -- python imports --
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel,delayed

# -- project imports --
from pyutils import write_pickle,read_pickle
from .parallel import ProgressParallel

def get_cache_name(cname):
    DIR = Path("./output/pretty_plots/stat_test_properties/")
    if not DIR.exists(): DIR.mkdir()
    fpath = DIR / f"{cname}.pkl"
    return fpath

def load_cache(cname):
    fpath = get_cache_name(cname)
    if fpath.exists():
        sims = read_pickle(fpath)
        print(f"Loaded from cache: [{fpath}]")
    else: return None
    return sims

def store_cache(sims,cname):
    fpath = get_cache_name(cname)
    write_pickle(sims,fpath)
    print(f"Wrote to cache: [{fpath}]")

def filter_complete_exps(pgrid,sims):

    def filter_pandas_by_dict(df,pydict):
        compare = df[list(pydict)] == pd.Series(pydict)
        loc = compare.all(1)
        return df.loc[loc]

    def filter_complete_exps_parallel(sims,i,_p):
        p = copy.deepcopy(_p)
        del p['size']
        fsims = filter_pandas_by_dict(sims,p)
        if len(fsims) == 0: return i
        else: return -1

    print("Filtering Complete Experiments from Todo.")
    if sims is None: return pgrid

    keep_index = []
    pParallel = ProgressParallel(True,len(pgrid),n_jobs=8)
    # pParallel = Parallel(n_jobs=8)
    delayed_fxn = delayed(filter_complete_exps_parallel)
    keep_index = pParallel(delayed_fxn(sims,i,_p) for i,_p in enumerate(pgrid))
    keep_index = sorted(np.unique(keep_index))[1:] # remove -1
    f_pgrid = [pgrid[i] for i in keep_index]

    return f_pgrid

def filter_grid_field(grid,field):
    new_grid = copy.deepcopy(grid)
    for idx in range(len(new_grid)):
        del new_grid[idx][field]
    return new_grid
