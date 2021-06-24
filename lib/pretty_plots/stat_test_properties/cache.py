
# -- python imports --
from pathlib import Path

# -- project imports --
from pyutils import write_pickle,read_pickle


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
