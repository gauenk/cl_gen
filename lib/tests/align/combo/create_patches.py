

# -- python --
import numpy as np
import numpy.random as npr

def patches_v1(T,B,C,H,W,eps,std):
    clean = npr.normal(loc=0,scale=eps,size=(T,B,C,H,W))
    noisy = npr.normal(loc=clean,scale=std)
    return clean,noisy
