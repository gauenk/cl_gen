

# -- lib imports --
import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")
import matplotlib
matplotlib.use('Agg')

# -- experiment package imports --
import noisy_alignment
import unsup_denoising_dl
import unsup_denoising_cl
import sup_denoising
import noisy_hdr
import picker

def run():
    # noisy_alignment.run()
    picker.run()

if __name__ == "__main__":
    run()
