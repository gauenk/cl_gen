
# -- python imports --
import numpy as np
from pathlib import Path


def get_tensor_cache_names(path,data):
    """
    Read "names.npy" 

    .../path/
        names.npy [modify]
        names1.npy
        names2.npy
        ...
        namesM.npy
        results.pkl [optional]

    data is
    1.) a torch.Tensor
    2.) a dictionary of torch.Tensor

    """
    names_fn = path / "names.npy"
    if not names_fn.exists():
        if isinstance(data,dict): names = np.array(list(data.keys))
        else: names = None
    else:
        names = np.load(names_fn,allow_pickle=False)        
    return names

