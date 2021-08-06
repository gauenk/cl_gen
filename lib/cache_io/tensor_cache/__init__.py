"""
Save a Python-dict of (a) tensors and/or (b) Python-dict-of-tensor-elements



"""

from ._write import *
from ._read import *
from ._append import *
from ._convert import *

class TensorCache():

    def __init__(self,root):
        self.root = root

    #------------------------------------
    #        Primary Functions
    #------------------------------------

    def convert_files_to_tensors(self,uuid,results):
        path = self.root / uuid # used to just be self.root
        return convert_files_to_tensors(path,results)

    def convert_tensors_to_files(self,uuid,results):
        path = self.root / uuid # used to just be self.root
        return convert_tensors_to_files(path,results)

    #------------------------------------
    #         Helper Functions
    #------------------------------------

    def append_tensor_cache(self,path,data,dim=2,overwrite=False):
        return append_tensor_cache(path,data,dim,overwrite)
        
    def write_tensor_cache(self,path,data):
        return write_tensor_cache(path,data)
        
    def write_dict_tensor_cache(self,path,data):
        return write_dict_tensor_cache(path,data)

    def read_tensor_cache(self,path,names):
        return read_tensor_cache(path,names)

    def read_dict_tensor_cache(self,path,names):
        return read_dict_tensor_cache(path,names)
        
    def get_tensor_cache_names(self,path,data):
        return get_tensor_cache_names(path,data)

