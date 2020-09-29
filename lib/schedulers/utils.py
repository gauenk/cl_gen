import math
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import MultiStepLR
from .linear_warmup import LinearWarmup

def get_simclr_scheduler(optimizer,batch_size,epochs,burnin,batches_per_epoch,load_epoch=-1):

    # grab lr from re-loaded optim
    init_lr = [group['lr'] for group in optimizer.param_groups]    

    # correct learning rate with batch size
    batch_correction = math.sqrt(batch_size)
    
    # compute number of total batches
    burnin_steps = int(burnin * batches_per_epoch)
    total_steps = int(epochs * batches_per_epoch)

    # convert epoch number to batches
    nbatches = load_epoch * batches_per_epoch if load_epoch > -1 else -1
    loading_lin = nbatches > -1

    # compute the loaded "epoch" cosine scheduler
    cos_batch = nbatches - burnin_steps if nbatches > -1 else -1
    cos_batch = cos_batch - 2 if cos_batch > -1 else -1 

    # init cosine annealing scheduler
    T_max = total_steps - burnin_steps
    eta_min = 1e-8
    after_scheduler = CosineAnnealingLR(optimizer,T_max,eta_min,cos_batch)
    cos_loading = cos_batch > -1

    # init the linear warmup
    nbatches = nbatches - 1 if nbatches > 0 else -1
    scheduler = LinearWarmup(optimizer, burnin_steps, batch_correction,
                             after_scheduler, nbatches)

    # handle setting lr after reloading
    if loading_lin:
        for idx,group in enumerate(optimizer.param_groups):
            group['lr'] = init_lr[idx]

    if cos_loading:
        after_scheduler._last_lr = init_lr
        scheduler._last_lr = init_lr

    return scheduler


def linear_warmup_reduce_on_palteau(optimizer,batch_size,epochs,burnin,batches_per_epoch,load_epoch=-1,mode="min"):

    # grab lr from re-loaded optim
    init_lr = [group['lr'] for group in optimizer.param_groups]    

    # correct learning rate with batch size
    batch_correction = math.sqrt(batch_size)
    
    # compute number of total batches
    burnin_steps = int(burnin * batches_per_epoch)
    total_steps = int(epochs * batches_per_epoch)

    # convert epoch number to batches
    nbatches = load_epoch * batches_per_epoch if load_epoch > -1 else -1
    loading_lin = nbatches > -1

    # compute the loaded "epoch" cosine scheduler
    cos_batch = nbatches - burnin_steps if nbatches > -1 else -1
    cos_batch = cos_batch - 2 if cos_batch > -1 else -1 

    # init cosine annealing scheduler
    T_max = total_steps - burnin_steps
    eta_min = 0
    after_scheduler = ReduceLROnPlateau(optimizer,
                                        patience = 10,
                                        factor=1./np.sqrt(10),
                                        mode=mode)
    cos_loading = cos_batch > -1

    # init the linear warmup
    nbatches = nbatches - 1 if nbatches > 0 else -1
    scheduler = LinearWarmup(optimizer, burnin_steps, batch_correction,
                             after_scheduler, nbatches)

    # handle setting lr after reloading
    if loading_lin:
        for idx,group in enumerate(optimizer.param_groups):
            group['lr'] = init_lr[idx]

    if cos_loading:
        after_scheduler._last_lr = init_lr
        scheduler._last_lr = init_lr

    return scheduler


def linear_warmup_multistep(optimizer,batch_size,epochs,burnin,batches_per_epoch,load_epoch=-1):

    # grab lr from re-loaded optim
    init_lr = [group['lr'] for group in optimizer.param_groups]    

    # correct learning rate with batch size
    batch_correction = math.sqrt(batch_size)
    
    # compute number of total batches
    burnin_steps = int(burnin * batches_per_epoch)
    total_steps = int(epochs * batches_per_epoch)

    # convert epoch number to batches
    nbatches = load_epoch * batches_per_epoch if load_epoch > -1 else -1
    loading_lin = nbatches > -1

    # compute the loaded "epoch" cosine scheduler
    cos_batch = nbatches - burnin_steps if nbatches > -1 else -1
    cos_batch = cos_batch - 2 if cos_batch > -1 else -1 

    # init cosine annealing scheduler
    T_max = total_steps - burnin_steps
    eta_min = 0
    milestones = [60,80]
    after_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    cos_loading = cos_batch > -1

    # init the linear warmup
    nbatches = nbatches - 1 if nbatches > 0 else -1
    scheduler = LinearWarmup(optimizer, burnin_steps, batch_correction,
                             after_scheduler, nbatches)

    # handle setting lr after reloading
    if loading_lin:
        for idx,group in enumerate(optimizer.param_groups):
            group['lr'] = init_lr[idx]

    if cos_loading:
        after_scheduler._last_lr = init_lr
        scheduler._last_lr = init_lr

    return scheduler


def get_train_scheduler(scheduler):
    """
    Do we need to run scheduler.step() after each batch? 
    """
    t1 = torch.optim.lr_scheduler.OneCycleLR
    t2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    t3 = LinearWarmup
    ts = [t1,t2,t3]
    tany = False
    for ti in ts:
        tany = tany or isinstance(scheduler,ti)
    if tany:
        return scheduler
    else:
        return None
