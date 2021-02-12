
# -- pytorch imports --
import torch

# -- project imports --
from layers.ame_kpn.KPN import KPN as KPN_model,LossFunc
from layers.unet import UNetN_v2,UNet_n2n,UNet_Git
from layers import UNetGP

def load_model(cfg):

    # model = UNet_n2n(cfg.input_N,5)
    # cfg.color_cat = True

    # model = UNet_Git(3*cfg.input_N,3)
    # cfg.color_cat = True

    model = UNetGP(cfg.input_N,cfg.unet_channels)
    cfg.color_cat = False

    # model = UNetN_v2(cfg.input_N,cfg.unet_channels)
    return model

def load_model_fp(cfg,model,model_fp,rank):
    if cfg.use_ddp:
        map_location = {'cuda:%d' % 0, 'cuda:%d' % rank}
    else:
        map_location = 'cuda:%d' % rank
    print(f"Loading model filepath [{model_fp}]")
    state = torch.load(model_fp, map_location=map_location)
    model.load_state_dict(state)
    return model

def load_model_field(cfg,rank,model,field):
    model = model.to(rank)
    if cfg.load:
        fn = Path("checkpoint_{}.tar".format(cfg.epoch_num))
        model_fp = Path(cfg.model_path) / Path(field) / fn
        model = load_model_fp(cfg,model,model_fp,rank)
    # if cfg.use_ddp:
    #     model = DDP(model, device_ids=[rank])
    return model

def load_model_kpn(cfg):
    return KPN_model(color=True,burst_length=cfg.input_N,blind_est=True),LossFunc()
