"""
Loading and saving model parameters

"""

# python imports
from easydict import EasyDict as edict
from pathlib import Path

# pytorch imports
import torch
from torch import nn
from torch.nn import SyncBatchNorm

from apex.parallel import DistributedDataParallel as apex_DDP
from torch.nn.parallel import DistributedDataParallel as th_DDP

# project imports
from layers.denoising import DenoisingLoss,Encoder,Decoder,Projector,DecoderRes50,DecoderNoSkip,DecoderSimple
from layers.denoising import DenoisingBlock
from simcl.model_io import load_model as simcl_load_model

#
# loading models
#

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

def load_encoder(cfg,rank):
    nc = cfg.n_img_channels
    h_size = cfg.enc_size
    # model = Encoder(n_channels = nc, embedding_size = h_size, use_bn=cfg.use_bn)
    model = EncoderN(n_channels = nc, embedding_size = h_size, use_bn=cfg.use_bn)
    model = load_model_field(cfg,rank,model,"encoder")
    return model

def load_decoder(cfg,rank):
    nc = cfg.n_img_channels
    h_size = cfg.enc_size
    if cfg.simcl.load:
        model = Decoder(n_channels = nc, embedding_size = h_size, use_bn=cfg.use_bn)
    else:
        model = DecoderSimple(n_channels = nc, embedding_size = h_size, use_bn=cfg.use_bn) 

    # model = DecoderNoSkip(n_channels = nc, embedding_size = h_size, use_bn=cfg.use_bn)
    # model = DecoderRes50(n_channels = nc, embedding_size = h_size, use_bn=cfg.use_bn)
    # model = load_model_field(cfg,rank,model,"decoder")
    model = load_model_field(cfg,rank,model,"./")
    return model

def load_projector(cfg,rank):
    h_size = cfg.enc_size
    z_size = cfg.proj_size
    model = Projector(h_size,z_size)
    model = load_model_field(cfg,rank,model,"projector")
    return model

def load_models(cfg,rank,proc_group):
    models = edict()

    # HACK: just get the models loaded but parameters are
    # saved together in block
    load = cfg.load
    cfg.load = False
    if cfg.simcl.load:
        cfg.simcl.rank = rank
        cfg.simcl.device = cfg.device
        cfg.simcl.denoising_prep = True
        simcl = simcl_load_model(cfg.simcl,rank,proc_group)
        encoder = simcl.encoder
        projector = simcl.projector
        # encoder.eval()
        # projector.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        for param in projector.parameters():
            param.requires_grad = False
    else:
        encoder = load_encoder(cfg,rank)
        projector = load_projector(cfg,rank)
    decoder = load_decoder(cfg,rank)
    cfg.load = load
    model = DenoisingBlock(encoder,decoder,
                           projector,rank,cfg.N,cfg.batch_size,
                           cfg.agg_enc_fxn,cfg.agg_enc_type)
    # if cfg.load:
    #     load_model_field(cfg,rank,model,"")
    if cfg.use_ddp:
        if cfg.sync_batchnorm:
            fxn = SyncBatchNorm.convert_sync_batchnorm
            model = fxn(model,proc_group)
        if cfg.use_apex:
            model = apex_DDP(model)
        else:
            model = th_DDP(model,device_ids=[rank],find_unused_parameters=True)
    return model

