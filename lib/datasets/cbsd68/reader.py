
# -- python imports --
import glob
import numpy as np
from pathlib import Path

def read_split_ids(idir,split):
    return []

def read_files(idir,split,isize):

    # -- cropping --
    if not(isize is None):
        print("Warning: we don't use crops right now for cbsd68.")

    # -- read all files --
    paths = []
    idir_star = str(idir) + "*"
    for fn in glob.glob(idir_star):
        paths.append(fn)

    # -- read split ids --
    split_ids = read_split_ids(idir,split)

    # -- filter by ids --
    fpaths = []
    for path in paths:
        fpaths.append(path)
        # if path in split_ids:
        #     fpaths.append(path)
        
    return fpaths
