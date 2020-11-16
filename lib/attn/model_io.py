
# -- python imports --
from pathlib import Path

# -- pytorch imports --

# -- project imports --
from layers.attn import TransformerNetwork
from layers.simcl import ClBlock,Encoder,Projector
from simcl.model_io import load_model as simcl_load_model


def load_model(cfg):
    simclr = load_model_simclr(cfg)
    model = TransformerNetwork(simclr)
    model = model.to(cfg.device)
    return model

def load_model_simclr(cfg):

    # encoder = get_resnet("resnet50", cfg.dataset.name, False, False)
    # projector = Projector(2048, 64)
    # block = ClBlock(encoder,projector,cfg.device,cfg.N,cfg.batch_size)
    # block = block.to(cfg.device)
    # return block

    proc_group = None
    rank = 0
    cfg.simcl.load = True
    cfg.simcl.rank = rank
    cfg.simcl.device = cfg.device
    cfg.simcl.denoising_prep = True
    simcl = simcl_load_model(cfg.simcl,rank,proc_group)
    return simcl
