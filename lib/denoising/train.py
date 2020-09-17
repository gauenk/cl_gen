"""
Run the training loop for denoising experiment
"""


# python imports
import time
import numpy.random as npr

# pytorch imports
from torch.utils.tensorboard import SummaryWriter
from apex import amp

# project imports
from pyutils.timer import Timer
from pyutils.misc import get_model_epoch_info
from schedulers import get_train_scheduler
from layers.denoising import DenoisingLossDDP
from learning.train import thtrain_denoising as train_loop
from learning.test import thtest_denoising as test_loop
from learning.utils import save_model,save_optim

# local proj imports
from .model_io import load_models
from .optim_io import load_optimizer
from .scheduler_io import load_scheduler
from .config import load_cfg,save_cfg,get_cfg,get_args
from .utils import load_hyperparameters,extract_loss_inputs

def run_train(cfg,rank,models,data,loader):
    only_one_print = (rank == 0 and cfg.use_ddp) or (cfg.use_ddp is False)

    s = int(npr.rand()*5+1)
    time.sleep(s)

    hyperparams = load_hyperparameters(cfg)

    criterion_inputs = [hyperparams]
    criterion_inputs += extract_loss_inputs(cfg,rank)
    criterion = DenoisingLossDDP(*criterion_inputs)
    criterion = criterion.to(cfg.device)

    optimizer = load_optimizer(cfg,models)
    scheduler = load_scheduler(cfg,optimizer,len(loader.tr))

    # apply apex
    if cfg.use_apex:
        models, optimizer = amp.initialize(models, optimizer, opt_level='O2')

    # init writer
    if only_one_print:
        writer = SummaryWriter(filename_suffix=cfg.exp_name)
    else:
        writer = None

    # init training loop
    global_step,current_epoch = get_model_epoch_info(cfg)
    cfg.global_step = global_step
    cfg.current_epoch = current_epoch

    # training loop
    tr_scheduler = get_train_scheduler(scheduler)
    test_losses = {}

    print(f"cfg.epochs: {cfg.epochs}")
    print(f"cfg.use_apex: {cfg.use_apex}")
    print("len of loader.val", len(loader.val))
    t = Timer()
    for epoch in range(cfg.current_epoch, cfg.epochs):
        t.tic()
        loss_epoch = train_loop(cfg, loader.tr, models,
                                  criterion, optimizer, epoch, writer,
                                  tr_scheduler)
        t.toc()
        lr = optimizer.param_groups[0]["lr"]
        # print(t)

        # if scheduler and tr_scheduler is None:
        #     val_loss = test_loop(cfg,models,loader.val)
        #     scheduler.step(val_loss)

        if epoch % cfg.checkpoint_interval == 0 and only_one_print:
            save_denoising_model(cfg,models,optimizer)

        if epoch % cfg.test_interval == 0:
            if only_one_print:
                te_loss = test_loop(cfg,models,loader.te)
                writer.add_scalar("Loss/test", te_loss, epoch)

        if only_one_print:
            writer.add_scalar("Loss/train", loss_epoch / len(loader.tr), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            msg = f"Epoch [{epoch}/{cfg.epochs}]\t"
            msg += f"Loss: {loss_epoch / len(loader.tr)}\t"
            msg += "{:2.3e}".format(lr)
            print(msg)
        cfg.current_epoch += 1


    if only_one_print:
        te_loss = test_loop(cfg,models,loader.te)
        writer.add_scalar("Loss/test", te_loss, epoch)
    save_denoising_model(cfg,models,optimizer)

def save_denoising_model(cfg,model,optimizer):
    save_model(cfg, model, None)
    save_optim(cfg, optimizer)    


