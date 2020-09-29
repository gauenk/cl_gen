"""
Load models for Simple Contrastive Learning

"""

# python imports
from pathlib import Path

# pytorch imports
import torch
from torch import nn
from torch.nn import SyncBatchNorm

from apex.parallel import DistributedDataParallel as apex_DDP
from torch.nn.parallel import DistributedDataParallel as th_DDP

# project imports
from layers import get_resnet
from layers.simcl import ClBlock,Projector,Encoder

def load_model(cfg,rank,proc_group):

    # initialize ResNet model
    encoder = load_encoder(cfg,rank)
    projector = load_projector(cfg,rank)

    # initialize model
    model = ClBlock(encoder, projector, cfg.device, cfg.N, cfg.batch_size)

    if cfg.load:
        fn = Path("checkpoint_{}.tar".format(cfg.epoch_num))
        model_fp = cfg.model_path / fn
        if cfg.use_ddp:
            map_location = lambda storage, loc: {'cuda:%d' % 0, 'cuda:%d' % rank}
        else:
            map_location = lambda storage, loc: storage.cuda(rank)
        map_location = lambda storage, loc: storage.cuda(rank)
        model.load_state_dict(torch.load(model_fp, map_location=map_location))
    model = model.to(cfg.device)
    print("loading model: ",model.device)

    if cfg.use_ddp:
        if cfg.sync_batchnorm:
            fxn = SyncBatchNorm.convert_sync_batchnorm
            model = fxn(model,proc_group)
        if cfg.use_apex:
            model = apex_DDP(model)
        else:
            model = th_DDP(model,device_ids=[cfg.rank],
                           find_unused_parameters=True)

    return model

def load_encoder(cfg,rank):
    nc = cfg.n_img_channels
    h_size = cfg.enc_size
    # model = get_resnet(cfg.resnet, cfg.dataset.name, pretrained=False)

    # workaround for cached exp_set_v1
    if 'encoder_type' not in cfg.keys():
        cfg.encoder_type = 'simple'

    if cfg.encoder_type == "simple":
        model = Encoder(n_channels = nc, embedding_size = h_size, use_bn=cfg.use_bn)
    elif cfg.encoder_type == "resnet18" or cfg.encoder_type == "resnet50":
        model = get_resnet(cfg.encoder_type, cfg.dataset.name, pretrained=False)
    else:
        raise ValueError(f"Uknown encoder type [{cfg.encoder_type}]")
    # model = load_model_field(cfg,rank,model,"encoder")
    return model

def load_projector(cfg,rank):
    h_size = cfg.enc_size
    z_size = cfg.proj_size
    model = Projector(h_size,z_size)
    # model = load_model_field(cfg,rank,model,"projector")
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

