"""
Loading and saving scheduler parameters

"""

# python imports
import torch
import numpy as np

# project imports
from schedulers.utils import linear_warmup_reduce_on_palteau,linear_warmup_multistep,get_simclr_scheduler


def load_scheduler(cfg,optimizer,batches_per_epoch):

    # 
    # optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9, dampening=0, nesterov=True)
    # if cfg.disent.load:
    #     fn = Path("checkpoint_{}.tar".format(cfg.disent.epoch_num))
    #     optim_fn = Path(cfg.disent.optim_path) / fn
    #     optimizer.load_state_dict(torch.load(optim_fn, map_location=cfg.disent.device.type))
    # milestones = [400]
    milestones = [50,150]
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-3,
    #                                                 steps_per_epoch=tr_batches,
    #                                                 epochs=cfg.disent.epochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    if cfg.test_with_psnr: mode = 'max'
    else: mode = 'min'
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                        mode=mode,
    #                                                        patience = 10,
    #                                                        factor=1./np.sqrt(10))
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    burnin = 10 # cfg.sched_params['burnin']
    if cfg.load:
        load_epoch = cfg.epoch_num
    else:
        load_epoch = -1
    scheduler = get_simclr_scheduler(optimizer,batch_size,epochs,burnin,batches_per_epoch,load_epoch)
    # scheduler = linear_warmup_reduce_on_palteau(optimizer,batch_size,epochs,burnin,batches_per_epoch,load_epoch=-1,mode)
    # scheduler = linear_warmup_multistep(optimizer,batch_size,epochs,burnin,batches_per_epoch,load_epoch=-1)

    
    return scheduler

