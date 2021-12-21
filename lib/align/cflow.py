
# -- import cflow pacakge --
import sys
sys.path.append("/home/gauenk/Documents/packages/cflow/pylib/")
import cflow

# -- python imports --
import torch
import numpy as np
from einops import rearrange


def runBurst(burst):

    nframes,nimages,c,h,w = burst.shape
    ref = nframes//2
    burst = rearrange(burst,'t i c h w -> t i h w c')
    flows = []
    compute_of = cflow.OpticalFlow_ComputeOpticalFlow
    for i in range(nimages):
        flows_t = []
        for t in range(nframes):
            # -- skip ref --
            if t == ref:
                flows_t.append(torch.zeros_like(flows_t[-1]))
                continue

            # -- get data --
            im1 = burst[ref,i].cpu().numpy().astype(np.double)
            im2 = burst[t,i].cpu().numpy().astype(np.double)
            im1 = np.ascontiguousarray(im1)
            im2 = np.ascontiguousarray(im2)
            flow = np.zeros((h,w,2))
        
            # -- to swig --
            h,w,c = im1.shape
            im1_swig = cflow.swig_ptr(im1)
            im2_swig = cflow.swig_ptr(im2)
            flow_swig = cflow.swig_ptr(flow)

            # -- compute flow --
            compute_of(im1_swig,im2_swig,flow_swig,h,w,c)
            flows_t.append(torch.FloatTensor(flow))
        flows_t = torch.stack(flows_t)
        flows.append(flows_t)
    flows = torch.stack(flows)
    flows = rearrange(flows,'i t h w two -> i (h w) t two')

    return flows
