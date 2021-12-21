
# -- python imports --
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from einops import rearrange
from easydict import EasyDict as edict
import pprint
pp = pprint.PrettyPrinter(indent=4)

import sys
sys.path.append("/home/gauenk/Documents/faiss/contrib/")
from nnf_share import mode_ndarray


def main():
    # theory_gaussian_gamma()
    theory_grouped_ave()

def theory_grouped_ave():

    #
    # -- Data Params --
    #

    std = 25./255.
    B = 10000
    t = 3
    p = 27

    # 
    # -- Theory Eqs --
    #

    var = std**2
    theory = edict()
    theory.c2 = ((t-1)/t)**2 * var + (t-1)/t**2 * var
    theory.mean = theory.c2*p
    theory.mode = (1 - 2/p)*theory.c2*p
    theory.var = 2*p*theory.c2**2
    theory.std = np.sqrt(theory.var)
    pp.pprint(theory)

    #
    # -- Create Data --
    #

    # a = np.random.normal(0,std,size=(B))
    # b = np.random.normal(0,std,size=(B))
    # c = np.random.normal(0,std,size=(B))
    # d = (a+b+c)/3.
    # print(np.std(d))
    
    obs = np.random.normal(0,std,size=(B,t,p))
    ave = np.mean(obs,axis=1)[:,None]
    # print("a",np.std(ave))
    # print("b",np.std(obs[:,[0],:] - ave)**2)
    # print(np.std(ave)**2)
    # print(np.std(obs[:,[0],:] - t*ave)**2)
    deltas = np.sum((obs[:,[0],:] - ave)**2,axis=-1)
    deltas = rearrange(deltas,'b t -> (b t)')

    #
    # -- Results --
    #

    mode = mode_ndarray(deltas)
    print("F - Ave")
    print("Mean: ",np.mean(deltas))
    print("Std: ",np.std(deltas))
    print("Mode: ",mode)


def theory_gaussian_gamma():

    # -- data params --
    std = 25./255.
    B = 10000
    t = 3
    p = 27

    # -- theory eqs --
    theory = edict()
    theory.c2 = ((t-1)/t)**2 * std**2 + (t-1)/t**2 * std**2
    theory.mean = theory.c2
    theory.mode = (1 - 2/p)*theory.c2
    theory.var = np.sqrt(2/p*theory.c2**2)
    pp.pprint(theory)

    # -- create data --
    # a = np.random.normal(0,std,size=(B))
    # b = np.random.normal(0,std,size=(B))
    # c = np.random.normal(0,std,size=(B))
    # d = (a+b+c)/3.
    # print(np.std(d))
    
    obs = np.random.normal(0,std,size=(B,t,p))
    ave = np.mean(obs,axis=1)[:,None]
    print("a",np.std(ave))
    print("b",np.std(obs[:,[0],:] - ave)**2)
    # print("c",np.std((obs[:,[1],:] - obs[:,[0],:] + obs[:,[2],:]- obs[:,[0],:] + obs[:,[3],:] - obs[:,[0],:] + obs[:,[4],:] - obs[:,[0],:])/5.)**2)
    print(np.std(ave)**2)
    print(np.std(obs[:,[0],:] - t*ave)**2)
    deltas = np.mean((obs[:,[0],:] - ave)**2,axis=-1)
    deltas = rearrange(deltas,'b t -> (b t)')

    # -- results --
    print("F - R")
    delta_fr = (obs[1:] - obs[:-1])**2
    mode = mode_ndarray(delta_fr)
    print("Mean: ",np.mean(delta_fr))
    print("Std: ",np.std(delta_fr))
    print("Mode: ",mode)

    mode = mode_ndarray(deltas)
    print("F - Ave")
    print("Mean: ",np.mean(deltas))
    print("Std: ",np.std(deltas))
    print("Mode: ",mode)

    print(deltas.shape)
    # means = np.mean(deltas,axis=1)
    # stds = np.std(deltas,axis=1)
    # print("Mean: ",np.mean(means))
    # print("Std: ",np.mean(stds))

    # -- create plots --
    fig,ax = plt.subplots(1,figsize=(8,8))
    sns.kdeplot(deltas.ravel(),label="obs",ax=ax)
    plt.savefig("./bp_search_rand_noisy_align.png",
                transparent=True,dpi=300)
    plt.close("all")


if __name__ == "__main__":
    main()

