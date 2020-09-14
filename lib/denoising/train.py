"""
Run the training loop for denoising experiment
"""


# python imports

# pytorch imports
from torch.utils.tensorboard import SummaryWriter
from apex import amp

# project imports
from pyutils.timer import Timer
from pyutils.misc import get_model_epoch_info
from schedulers import get_train_scheduler
from layers.denoising import DenoisingLossDDP
from learning.train import thtrain_denoising as train_loop
# from learning.test import thtest_denoising as test_loop


# local proj imports
from .model_io import load_models
from .optim_io import load_optimizer
from .scheduler_io import load_scheduler
from .config import load_cfg,save_cfg,get_cfg,get_args
from .utils import load_hyperparameters,extract_loss_inputs

def run_train(cfg,rank,models,data,loader):

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
    writer = SummaryWriter(filename_suffix=cfg.exp_name)

    # init training loop
    global_step,current_epoch = get_model_epoch_info(cfg)
    cfg.global_step = global_step
    cfg.current_epoch = current_epoch

    # training loop
    tr_scheduler = get_train_scheduler(scheduler)
    test_losses = {}

    t = Timer()
    for epoch in range(cfg.current_epoch, cfg.epochs+1):
        lr = optimizer.param_groups[0]["lr"]

        t.tic()
        loss_epoch = train_loop(cfg, loader.tr, models,
                                  criterion, optimizer, epoch, writer,
                                  tr_scheduler)
        t.toc()
        # print(t)

        # if scheduler and tr_scheduler is None:
        #     val_loss = test_loop(cfg,models.enc_c,models.dec,loader.val)
        #     scheduler.step(val_loss)

        # if epoch % cfg.checkpoint_interval == 0:
        #     save_disent_models(cfg,models,optimizer)

        # if epoch % cfg.test_interval == 0:
        #     te_loss = test_static(cfg,models.enc_c,models.dec,loader.te)
        #     writer.add_scalar("Loss/test", te_loss, epoch)

        writer.add_scalar("Loss/train", loss_epoch / len(loader.tr), epoch)
        writer.add_scalar("Misc/learning_rate", lr, epoch)
        print(
            f"Epoch [{epoch}/{cfg.epochs}]\t Loss: {loss_epoch / len(loader.tr)}\t " + "{:2.3e}".format(lr)
        )
        cfg.current_epoch += 1

    # save_disent_models(cfg,models,optimizer)
