
"""

The object in memory to inferace with ExpCache

each field is a tensor or dictionary of tensors

"""

import shutil,os
from pathlib import Path
from .impl import format_tensor_results,stack_ndarray

class ExpRecord():

    def __init__(self,dims=None):
        self.record = {}
        self.batch_record = {}
        self.dims = self._init_dims_dict(dims)

    def clear_batch_record(self):
        self.batch_record = {}

    def append_batch_results(self,inputs,dims=None):
        if dims is None: dims = self.dims['batch_results']
        return format_tensor_results(inputs,self.batch_record,dims,append=True)

    def append(self,inputs,dims=None):
        return self.append_record_results(inputs,dims)

    def append_record_results(self,inputs,dims=None):
        # inputs = scalar_to_tensor(inputs)
        if dims is None: dims = self.dims['record_results']
        return format_tensor_results(inputs,self.record,dims,append=True)

    def append_batch_to_record(self,inputs,dims=None):
        if dims is None: dims = self.dims['batch_to_record']
        return format_tensor_results(self.batch_record,self.record,dims,append=True)
    
    def cat_record(self,dims=None):
        if dims is None: dims = self.dims['cat']
        return format_tensor_results(self.record,self.record,dims,append=False)

    def stack_record(self,dims=None):
        if dims is None: dims = self.dims['stack']
        return stack_ndarray(self.record,dims)

    def print_record(self):
        for fieldname,results_f in self.results.items():
            if not isinstance(results_f,dict):
                print(results_f.shape)
                continue
            for sname,results_s in results_f.items():
                print(sname,results_s.shape)

    def _init_dims_dict(self,dims):
        keys = ['batch_results','record_results','batch_to_record','cat']
        if not(dims is None):
            for key in keys:
                assert key in dims, f"dims must have key [{key}]"
            return dims
        else:
            default = {key:{'default':0} for key in keys}
            return default
