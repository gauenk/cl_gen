

# -- lib imports --
import sys,os
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")
import matplotlib
matplotlib.use('Agg')

# -- experiment package imports --
import noisy_alignment
import unsup_denoising
import sup_denoising
import noisy_hdr
import picker

def run():
    print("PID: [{}]".format(os.getpid()))
    # unsup_denoising.run()
    noisy_alignment.run()
    # picker.run()

if __name__ == "__main__":
    run()
