"""
Run the testing loop for denoising experiment
"""
# python imports
from easydict import EasyDict as edict

# pytorch imports
from apex import amp

# local proj imports
from learning.test import thtest_denoising as test_loop

def run_test(cfg,rank,model,data,loader,n_runs=1):

    print(f"Testing image denoising with epoch {cfg.epoch_num}")
    # apply apex
    if cfg.use_apex:
        model = amp.initialize(model, opt_level='O2')

    te_losses = []
    for n in range(n_runs):
        te_loss = test_loop(cfg,  model, loader.te)
        te_losses.append(te_loss)
    if n_runs > 1:
        mean = np.mean(te_losses)
        stderr = np.std(te_losses) / np.sqrt(len(te_losses))
    else:
        mean = te_losses[0]
        stderr = 0.

    # print("Testing loss: {:.3f}".format(tr_loss))
    # print("Testing loss: {:.3f}".format(val_loss))
    print("Testing loss: {:2.3e} +/- {:2.3e}\n".format(mean,1.96*stderr))
    losses = edict()
    losses.te_losses = te_losses
    losses.mean = mean
    losses.stderr = stderr
    return losses
