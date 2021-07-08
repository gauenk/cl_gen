
from .impl import convert_files_to_tensors,convert_tensors_to_files,append_tensor_cache,write_tensor_cache,write_dict_tensor_cache,get_tensor_cache_names,read_tensor_cache,read_dict_tensor_cache

class TensorCache():

    def __init__(self,root):
        pass

    def convert_files_to_tensors(self,root,results):
        return convert_files_to_tensors()

    def convert_tensors_to_files(self,root,results):
        return convert_tensors_to_files(root,results)

    def append_tensor_cache(self,root,fieldname,data,dim=2,overwrite=False):
        
    def write_tensor_cache(self,path,data):
        
    def write_dict_tensor_cache(self,path,data):
        
    def get_tensor_cache_names(self,path,data):
        
    def read_tensor_cache(self,path,names):
        
    def read_dict_tensor_cache(self,path,names):
        


    
