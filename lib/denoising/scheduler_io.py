"""
Loading and saving scheduler parameters

"""

import numpy as np
import torch

def load_scheduler(cfg,optimizer,batches_per_epoch):

    # 
    # optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9, dampening=0, nesterov=True)
    # if cfg.disent.load:
    #     fn = Path("checkpoint_{}.tar".format(cfg.disent.epoch_num))
    #     optim_fn = Path(cfg.disent.optim_path) / fn
    #     optimizer.load_state_dict(torch.load(optim_fn, map_location=cfg.disent.device.type))
    # milestones = [400]
    # milestones = [75,150,350]
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-3,
    #                                                 steps_per_epoch=tr_batches,
    #                                                 epochs=cfg.disent.epochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience = 10,
                                                           factor=1./np.sqrt(10))
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    burnin = 10 # cfg.sched_params['burnin']
    load_epoch = -1 # cfg.epoch_num
    # scheduler = get_simclr_scheduler(optimizer,batch_size,epochs,burnin,batches_per_epoch,load_epoch)
    return scheduler

