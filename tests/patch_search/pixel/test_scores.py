
import torch
from patch_search import get_score_function

# def test_bootstrap(benchmark):
    
#     npatches = 1
#     nimages = 3
#     naligns = 100
#     nframes = 3
#     ncolors = 3
#     H,W = 15,15

#     data = torch.rand(npatches,nimages,
#                       naligns,nframes,
#                       ncolors,H,W)
    
#     bootstrapping = get_score_function("bootstrapping")
#     benchmark(bootstrapping,None,data)

# def test_bootstrap_mod1(benchmark):
    
#     npatches = 1
#     nimages = 3
#     naligns = 100
#     nframes = 3
#     ncolors = 3
#     H,W = 15,15

#     data = torch.rand(npatches,nimages,
#                       naligns,nframes,
#                       ncolors,H,W)
#     data = data.to('cuda:0')
    
#     bootstrapping = get_score_function("bootstrapping_mod1")
#     benchmark(bootstrapping,None,data)

def test_bootstrap_mod2(benchmark):
    
    npatches = 1
    nimages = 3
    naligns = 100
    nframes = 3
    ncolors = 3
    H,W = 15,15

    data = torch.rand(npatches,nimages,
                      naligns,nframes,
                      ncolors,H,W)
    data = data.to('cuda:0')
    
    bootstrapping = get_score_function("bootstrapping_mod2")
    benchmark(bootstrapping,None,data)


def test_ave(benchmark):
    
    npatches = 1
    nimages = 3
    naligns = 100
    nframes = 3
    ncolors = 3
    H,W = 15,15

    data = torch.rand(npatches,nimages,
                      naligns,nframes,
                      ncolors,H,W)
    data = data.to('cuda:0')
    
    ave = get_score_function("ave")
    benchmark(ave,None,data)

