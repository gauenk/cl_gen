
"""
Load the flownet model.

I am hoping I won't trigger an import error unless this function is called.
"""

import torch
from easydict import EasyDict as edict
from einops import rearrange

try:
    from layers.flownet_v2 import FlowNet2
except:
    print("No flownetv2 model. Not available for alignment.")

class ModFlowNet2(FlowNet2):
    def __init__(self,args,batchNorm=False,div_flow=20.):
        super().__init__(args,batchNorm=False,div_flow=20.)

    def burst2flow(self,burst):
        nframes = burst.shape[0]
        ref_t = nframes//2
        ref_frame = burst[ref_t]
        flows = []
        for t in range(nframes):
            if t == ref_t:
                flow_t = torch.zeros_like(flows[-1])
                flow_t = flow_t.to(flows[-1].device,non_blocking=True)
                flows.append(flow_t)
                continue
            frame_t = burst[t]
            inputs_t = torch.stack([ref_frame,frame_t],dim=-3)
            with torch.no_grad():
                flow_t = self(inputs_t)
            flows.append(flow_t)
        flows = torch.stack(flows,dim=0)
        flows = rearrange(flows,'t i two h w -> t i h w two')
        # nframes,nimages,h,w,two = flows.shape
        return flows
    
def load_flownet_model(cfg):
    args = edict({'rgb_max':1.,'fp16':False})
    model = ModFlowNet2(args,batchNorm=False,div_flow=20.).to(cfg.gpuid)
    state = torch.load("./checkpoints/FlowNet2_checkpoint.pth.tar")
    model.load_state_dict(state["state_dict"])
    return model
    
