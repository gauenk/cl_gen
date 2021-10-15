

# -- lib imports --
import sys,os
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")
import matplotlib
matplotlib.use('Agg')

# -- fix weird colorama error when using multiple threads --
import colorama
colorama.init() 

# -- experiment package imports --
import noisy_alignment
import unsup_denoising
import sup_denoising
import noisy_hdr
import picker

# -- torch multiprocessing for CUDA --
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

def run():
    print("PID: [{}]".format(os.getpid()))
    # unsup_denoising.run()
    # sup_denoising.run()
    noisy_alignment.run()
    # picker.run()

if __name__ == "__main__":
    run()
