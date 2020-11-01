
# python imports
from pathlib import Path

# pytorch imports

# project imports
from layers.resnet import get_resnet
from layers.unet import UNet,UNet_v2,UNet_v2_with_noise
from layers.simcl import ClBlock,Encoder,Projector
from layers.denoise_gan import Discriminator,Generator,DCGAN_D,DCGAN_G,DiscriminatorSimCLR, Discriminator_wp
from simcl.model_io import load_model as simcl_load_model


def load_model_noise(cfg):
    # model = Generator(100,32,3)
    # model = DCGAN_G(32,100,3,64,1,0)
    # model = UNet(3)
    model = UNet_v2_with_noise(3,False)
    model = model.to(cfg.device)
    return model

def load_model_rec(cfg):
    # model = Generator(100,32,3)
    # model = DCGAN_G(32,100,3,64,1,0)
    # model = UNet(3)
    model =UNet_v2(3,False)
    model = model.to(cfg.device)
    return model

def load_model_gen(cfg):
    # model = Generator(100,32,3)
    # model = DCGAN_G(32,100,3,64,1,0)
    # model = UNet(3)
    model =UNet_v2(3,False)
    model = model.to(cfg.device)
    return model

def load_model_disc(cfg,use_simclr=False):
    if use_simclr:
        model = DiscriminatorSimCLR(32,2,64,1,0)
    else:
        model = DCGAN_D(32,100,3,64,1,0)
        # model = Discriminator(32,3)
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
