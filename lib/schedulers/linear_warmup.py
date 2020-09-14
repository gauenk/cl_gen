import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings

class LinearWarmup(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_epoch: (epochs = batches here) target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, burnin_steps, batch_correction, after_scheduler, last_epoch=-1):
        self.burnin_steps = burnin_steps
        self.batch_correction = batch_correction
        self.after_scheduler = after_scheduler
        self.finished = last_epoch > burnin_steps
        self.loading = last_epoch >= 0
        self._init_step = True
        super(LinearWarmup, self).__init__(optimizer,last_epoch)
        self._init_step = False

    def state_dict(self):
        """
        Overwrite base class so we don't save after_scheduler either.
        """
        return {key: value for key, value in self.__dict__.items() if key not in ['optimizer','after_scheduler']}

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        batch_correction = self.batch_correction
        warmup_scaling = 1.0 if self.finished else float(self.last_epoch) / self.burnin_steps
        warmup_scaling = 1.0 if warmup_scaling > 1 else warmup_scaling
        scaled_lr = [warmup_scaling * base_lrs *  batch_correction for base_lrs in self.base_lrs]

        if self.last_epoch > self.burnin_steps and not self.finished:
            self.after_scheduler.base_lrs = scaled_lr
            self.finished = True
            self.after_scheduler.step()
            return self.after_scheduler.get_last_lr()
        elif self.finished:
            raise ValueError("We shouldn't call this function is self.finised is True")
        else:
            return scaled_lr

    def step(self, epoch=None, metrics=None):
        if self._init_step and self.loading:
            self.optimizer._step_count = self.last_epoch + 1
        if self.finished:
            self.after_scheduler.step()
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(LinearWarmup, self).step()
