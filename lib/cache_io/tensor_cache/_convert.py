import numpy as np
from pathlib import Path

from ._read import *
from ._append import *
from ._utils import *

def convert_files_to_tensors(root,results):
    tensors = {}
    for field,field_data in results.items():
        path =  root / field
        names = get_tensor_cache_names(path,field_data)
        data = read_tensor_cache(path,names)
        tensors[field] = data
    return tensors

def convert_tensors_to_files(root,results):
    files = {}
    for field,field_data in results.items():
        path = root / field
        append_tensor_cache(path,field_data,overwrite=True)
        files[field] = str(path)
    return files
