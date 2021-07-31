
# -- python --
from pathlib import Path
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- pytorch --
import torch

# -- project --
from settings import ROOT_PATH
from layers.flownet_v2 import FlowNet2

class FlownetExperiment():

    def __init__(self,cfg):
        self.name = cfg.flownet_name
        self.rel_model_path = Path("./output/explore/flownet/")
        self.model_dir_path = Path(ROOT_PATH) / self.rel_model_path
        self.gpuid = cfg.gpuid
        # self.noise_grid = ['none','g-25','g-75','p-4','p-10','qis-4-3-25p0']
        self.noise_grid = ['none']
        self.models,self.paths = self.setup(cfg.flownet_name)

    def setup(self,name):
        flownet,f_version,m_version = name.split("-")
        if f_version != "v2": raise ValueError(f"Unknown flownet version [{version}]")
        model_dir = self.model_dir_path / f_version / m_version
        print(f"Loading flownet model from [{model_dir}]")
        models,paths = [],[]
        for model_noise in self.noise_grid:
            model_path = model_dir / model_noise / "model.pth.tar"
            model = load_flownet_model(f_version,model_path,self.gpuid)
            models.append(model),paths.append(model_path)
        return models,paths

    def run(self,cfg,clean,noisy,directions,results):
        run_flownet(self.models,self.noise_grid,cfg,clean,noisy,directions,results)

def load_flownet_model(version,model_path,default_gpuid):
    if version == "v2":
        default_params = {'args':edict({'fp16':False,'rgb_max':1.0}),
                          'batchNorm':False,
                          'div_flow':20.0}
        model = FlowNet2(**default_params)
    else:
        raise ValueError(f"Uknown Flownet version [{version}]")
    if model_path.exists():
        model = load_model_fp(model,model_path,default_gpuid)
    else:
        print(f"WARNING: no flownet model loaded at [{model_path}]")
    model = model.cuda(device=default_gpuid)
    model.eval()
    return model

def load_model_fp(model,model_fp,gpuid):
    # map_location = 'cuda:%d' % gpuid
    map_location = 'cpu'
    print(f"Loading model filepath [{model_fp}]")
    state = torch.load(model_fp, map_location=map_location)
    model.load_state_dict(state['state_dict'])
    return model
        
def run_flownet_model(cfg,model,noisy):
    # -- shapes --
    T,B,C,PS,PS = noisy.shape # input shape
    t_ref = T//2

    # -- pairs with ref  --
    ref = repeat(noisy[t_ref],'b c h w -> t b c h w',t=T-1)
    nonref = torch.cat([noisy[:t_ref],noisy[t_ref+1:]],dim=0)
    pairs = torch.stack([ref,nonref],dim=0) # 2, T-1, B, C, PS, PS
    pairs = rearrange(pairs,'p tm1 b c h w -> (tm1 b) c p h w')

    # -- fwd --
    flows = []
    PB = pairs.shape[0]
    with torch.no_grad():
        for b in range(PB):
            # print(b,pairs[[b]].shape)
            flow = model(pairs[[b]])
            flows.append(flow)
        flows = torch.cat(flows,dim=0)

    # -- reshape --
    flows = rearrange(flows,'(tm1 b) p h w -> tm1 b p h w',b=B)
    return flows

def run_flownet(models,noise_info,cfg,clean,noisy,gt_flow,results):
    T,B,C,PS,PS = clean.shape # input shape
    """
    P = different patches from same image ("R" in patch_match/pixel/score.py)
    B = different images
    T = burst of frames along a batch dimension
    """
    gt_flow = rearrange(gt_flow,'b tm1 p -> tm1 b p').cpu()
    results['gt_flow'] = [gt_flow]
    for model,model_noise in zip(models,noise_info):
        flows = run_flownet_model(cfg,model,noisy)
        flows = flows.cpu()
        fieldname = f'flownet_{model_noise}'
        results[fieldname] = {'flows':flows}

