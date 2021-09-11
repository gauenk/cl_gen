"""
Apply NVOF functions with some wrapper code 
for concise function calls

"""

import cv2
import numpy as np
from einops import rearrange
import torch


def nvof_burst(burst):

    burst = burst.cpu()
    nframes,nimages,c,h,w = burst.shape
    ref_t = nframes//2
    flows = []
    cuda_nvof = cv2.optflow.DualTVL1OpticalFlow_create()
    for i in range(nimages):
    
        nd_clean = rearrange(burst[:,i].numpy(),'t c h w -> t h w c')
        flows_i = []
        for t in range(nframes):
    
            if t == ref_t:
                # frames.append(nd_clean[t][None,:])
                flows_i.append(torch.zeros(flows_i[-1].shape))
                continue
            from_frame = 255.*cv2.cvtColor(nd_clean[ref_t],cv2.COLOR_RGB2GRAY)
            to_frame = 255.*cv2.cvtColor(nd_clean[t],cv2.COLOR_RGB2GRAY)
            _flow = cuda_nvof.calc(to_frame,from_frame,None)
            # _flow = cv2.calcOpticalFlowFarneback(to_frame,from_frame,None,
            #                                      0.5,3,3,10,5,1.2,0)
            _flow = np.round(_flow).astype(np.float32)
            _flow[...,0] = -_flow[...,0] # my OF is probably weird.
            # w_frame = warp_flow(nd_clean[t], -_flow)
            # print("w_frame.shape ",w_frame.shape)
            flows_i.append(torch.FloatTensor(_flow))
            # frames.append(torch.FloatTensor(w_frame[None,:]))
        flows_i = torch.stack(flows_i)
        flows_i = rearrange(flows_i,'t h w two -> (h w) t two')
        flows.append(flows_i)

    flows = torch.stack(flows)

    return flows
