"""
Loading and saving model parameters

"""

# python imports
from easydict import EasyDict as edict

# pytorch imports
import torch
from torch import nn
from torch.nn import SyncBatchNorm

from apex.parallel import DistributedDataParallel as apex_DDP
from torch.nn.parallel import DistributedDataParallel as th_DDP

# project imports
from layers.denoising import DenoisingLoss,Encoder,Decoder,Projector
from layers.denoising import DenoisingBlock

#
# loading models
#

def load_model_fp(cfg,model_fp,rank):
    if cfg.use_ddp:
        map_location = {'cuda:%d' % 0, 'cuda:%d' % rank}
    else:
        map_location = {'cuda:%d' % rank}
    state = torch.load(model_fp, map_location=map_location)
    model.load_state_dict(state)
    return model

def load_model_field(cfg,rank,model,field):
    model = model.to(rank)
    if cfg.load:
        fn = Path("checkpoint_{}.tar".format(cfg.epoch_num))
        model_fp = Path(cfg.model_path) / Path(field) / fn
        model = load_model_fp(cfg,model_fp,rank)
    # if cfg.use_ddp:
    #     model = DDP(model, device_ids=[rank])
    return model

def load_encoder(cfg,rank):
    nc = cfg.n_img_channels
    h_size = cfg.enc_size
    model = Encoder(n_channels = nc, embedding_size = h_size)
    model = load_model_field(cfg,rank,model,"encoder")
    return model

def load_decoder(cfg,rank):
    nc = cfg.n_img_channels
    h_size = cfg.enc_size
    model = Decoder(n_channels = nc, embedding_size = h_size)
    model = load_model_field(cfg,rank,model,"decoder")
    return model

def load_projector(cfg,rank):
    h_size = cfg.enc_size
    z_size = cfg.proj_size
    model = Projector(h_size,z_size)
    model = load_model_field(cfg,rank,model,"projector")
    return model

def load_models(cfg,rank,proc_group):
    models = edict()
    models.encoder = load_encoder(cfg,rank)
    models.decoder = load_decoder(cfg,rank)
    models.projector = load_projector(cfg,rank)
    model = DenoisingBlock(models.encoder,models.decoder,
                   models.projector,rank,cfg.N,cfg.batch_size,
                   cfg.agg_enc_fxn,cfg.agg_enc_type)
    if cfg.use_ddp:
        if cfg.sync_batchnorm:
            fxn = SyncBatchNorm.convert_sync_batchnorm
            model = fxn(model,proc_group)
        if cfg.use_apex:
            model = apex_DDP(model)
        else:
            model = th_DDP(model,device_ids=[rank],find_unused_parameters=True)
    return model

