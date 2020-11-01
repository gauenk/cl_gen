"""
Run the training loop for simcl experiment

thtest_simcl

load_optimizer
load_scheduler
"""


# python imports
import time,functools
import numpy as np
import numpy.random as npr
from pathlib import Path
import datetime
import multiprocessing as mp

# pytorch imports
from torch.utils.tensorboard import SummaryWriter
from apex import amp

# project imports
import settings
from pyutils.timer import Timer
from pyutils.misc import get_model_epoch_info
from schedulers import get_train_scheduler as get_loop_scheduler
from learning.train import thtrain_simcl as train_loop
from learning.test_simcl import thtest_simcl as test_loop
from learning.utils import save_model,save_optim
from layers import ClBlockLoss

# local proj imports
from .model_io import load_model
from .optim_io import load_optimizer
from .scheduler_io import load_scheduler
from .utils import load_hyperparameters
from .exp_utils import _build_v1_summary,_build_v2_summary

def run_train(cfg,rank,model,data,loader):
    this_proc_prints = (rank == 0 and cfg.use_ddp) or (cfg.use_ddp is False)

    s = int(npr.rand()*5+1)
    time.sleep(s)

    hyperparams = load_hyperparameters(cfg)

    criterion_inputs = [hyperparams]
    criterion = ClBlockLoss(hyperparams,cfg.N,cfg.batch_size)
    criterion = criterion.to(cfg.device)

    optimizer = load_optimizer(cfg,model)
    scheduler = load_scheduler(cfg,optimizer,len(loader.tr))
    print("Loaded optimizer: ")
    print(optimizer)
    print("Loaded scheduler: ")
    print(scheduler)

    # apply apex
    if cfg.use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    # init writer
    if this_proc_prints:
        datetime_now = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        writer_dir = cfg.summary_log_dir / Path(datetime_now)
        writer = SummaryWriter(writer_dir)
    else:
        writer = None

    # init training loop
    global_step,current_epoch = get_model_epoch_info(cfg)
    cfg.global_step = global_step
    cfg.current_epoch = current_epoch

    # training loop
    loop_scheduler = get_loop_scheduler(scheduler)
    test_losses = {}

    # if this_proc_prints:
    #     spawn_split_eval(cfg,'val',writer,24)

    print(f"cfg.epochs: {cfg.epochs}")
    print(f"cfg.use_apex: {cfg.use_apex}")
    print(f"cfg.use_bn: {cfg.use_bn}")
    print("len of loader.val", len(loader.val))
    print(f"cfg.optim_type: {cfg.optim_type}")
    print(f"cfg.checkpoint_interval: {cfg.checkpoint_interval}")
    print(_build_v2_summary(cfg))
    t = Timer()
    for epoch in range(cfg.current_epoch, cfg.epochs):
        t.tic()
        loss_epoch = train_loop(cfg, loader.tr, model,
                                  criterion, optimizer, epoch, writer,
                                  loop_scheduler)
        t.toc()
        lr = optimizer.param_groups[0]["lr"]
        # print(t)

        # if ms_scheduler:
        #     ms_scheduler.step()


        if epoch % cfg.checkpoint_interval == 0 and this_proc_prints and epoch > 0:
            save_simcl_model(cfg,model,optimizer)

        if scheduler and loop_scheduler is None and epoch % cfg.val_interval == 0 and epoch > 0:
            val_loss = test_loop(cfg,model,'val')
            scheduler.step(val_loss)
            if this_proc_prints:
                writer.add_scalar("Loss/val", val_loss, epoch)
        elif epoch % cfg.val_interval == 0 and this_proc_prints and epoch > 0:
            if this_proc_prints:
                # spawn_split_eval(cfg,'val',writer,epoch)
                val_loss = test_loop(cfg,model,'val')
                writer.add_scalar("Loss/val", val_loss, epoch)

        if epoch % cfg.test_interval == 0 and epoch > 0:
            if this_proc_prints:
                # spawn_split_eval(cfg,'test',writer,epoch)
                te_loss = test_loop(cfg,model,'test')
                writer.add_scalar("Loss/test", te_loss, epoch)

        if this_proc_prints:
            writer.add_scalar("Loss/train", loss_epoch / len(loader.tr), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            msg = f"Epoch [{epoch}/{cfg.epochs}]\t"
            msg += f"Loss: {loss_epoch / len(loader.tr)}\t"
            msg += "{:2.3e}".format(lr)
            print(msg)
        cfg.current_epoch += 1


    if this_proc_prints:
        te_loss = test_loop(cfg,model,'test')
        writer.add_scalar("Loss/test", te_loss, epoch)
    save_simcl_model(cfg,model,optimizer)

def save_simcl_model(cfg,model,optimizer):
    print("Saving model and optim state")
    save_model(cfg, model, None)
    save_optim(cfg, optimizer)
    


def spawn_split_eval(cfg,split,writer,epoch,gpuid=2):
    
    # -- load the model --
    use_ddp = cfg.use_ddp 
    load = cfg.load
    epoch_num = cfg.epoch_num
    rank = cfg.rank
    device = cfg.device

    cfg.use_ddp = False
    cfg.load = True
    cfg.epoch_num = epoch
    cfg.rank = 2
    cfg.device = 'cuda:2'
    model = load_model(cfg,2,None)

    cfg.use_ddp = use_ddp
    cfg.load = load
    cfg.epoch_num = epoch_num
    cfg.rank = rank
    cfg.device = device


    # -- run process asynchronously --
    print("Launch!")
    def cb_func_raw(writer,epoch,loss):
        print(loss)
        writer.add_scalar(f"Loss/{split}",loss,epoch)
    cb_func = functools.partial(cb_func_raw,writer,epoch)
    def error_cb_func(msg):
        print(msg)
    inputs = [cfg,model,split,epoch,gpuid]
    pool = mp.Pool(processes = 1)
    pool.apply_async(run_split_eval,args = inputs, callback = cb_func,
                     error_callback = error_cb_func)
    print(pool)
    pool.close()
    pool.join()
    print(pool)

    # asyncio.async(run_split_eval,inputs)
    # run_split_eval(cfg,model,split,writer,epoch,gpuid=gpuid)

    
def run_split_eval(cfg,model,split,epoch,gpuid):
    loss = test_loop(cfg,model,split)
    print("fLoss/{split}",loss,flush=True)
    return loss
