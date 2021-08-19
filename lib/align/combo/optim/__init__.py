"""
Optimizers over the Combinatorial Search Space

"""

import align.combo.optim.v1 as v1
import align.combo.optim.v3 as v3
from align.combo.optim.launcher import run_burst,run_image_batch,run_pixel_batch

def get_optim_run(version):
    if version == "v1":
        return v1.run
    elif version == "v3":
        return v3.run
    else:
        raise ValueError(f"Uknown optimizer version [{version}]")
    

class AlignOptimizer():

    def __init__(self,version):
        self.version = version
        self._run_fxn = get_optim_run(version)

    def __call__(self,*args,**kwargs):
        return run_burst(self._run_fxn,*args,**kwargs)

    def run(self,*args,**kwargs):
        return run_burst(self._run_fxn,*args,**kwargs)

    def exec(self,*args,**kwargs):
        return run_burst(self._run_fxn,*args,**kwargs)


