""" Layer-wise adaptive rate scaling for SGD in PyTorch! """
import torch
from torch.optim.optimizer import Optimizer, required

# TODO: implement this!

class LAMB(Optimizer):
    r"""Implements layer-wise adaptive moments optimizer for batch training.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        eta (float, optional): LARS coefficient
        max_epoch: maximum training epoch to determine polynomial LR decay.

    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888

    Example:
        >>> optimizer = LAMB(model.parameters(), lr=0.1, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, params, max_epoch, lr=required, momentum=.9,
                 weight_decay=.0005, eta=0.001):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}"
                             .format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        self.prev_lr = []
        self.epoch = 0
        self.clamp_range = [0,1e11]
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay,
                        eta=eta, max_epoch=max_epoch)
        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step_misc(self, epoch=None, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            epoch: current epoch to calculate polynomial LR decay schedule.
                   if None, uses self.epoch and increments it.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta'] # not in Alg 1; c.f. eq 6
            lr = group['lr']
            max_epoch = group['max_epoch']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data

                p.add_(d_p, alpha=-group['lr'])
        return loss

    @torch.no_grad()
    def step(self, epoch=None, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            epoch: current epoch to calculate polynomial LR decay schedule.
                   if None, uses self.epoch and increments it.
        """
        self.curr_lr = []
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta'] # not in Alg 1; c.f. eq 6
            lr = group['lr']
            max_epoch = group['max_epoch']

            # Global LR computed on polynomial decay schedule
            decay = (1 - float(epoch) / max_epoch) ** 2
            global_lr = lr * decay

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data


                # add 1e-16 for numerical stability; nan issues
                weight_norm = torch.norm(p.data).clamp(*self.clamp_range)
                grad_norm = torch.norm(d_p).clamp(*self.clamp_range)

                # if either norm is zero, local_lr = 1.
                if weight_norm 

                # Compute local learning rate for this layer
                local_lr = eta * weight_norm / \
                    (grad_norm + weight_decay * weight_norm)

                # Update the momentum term
                actual_lr = local_lr * global_lr
                actual_lr = actual_lr.clamp(*self.clamp_range)

                if torch.isnan(actual_lr):
                    print("actual_lr IS NA")
                    print("local_lr: {:2.3e}".format(local_lr))
                    print("global_lr: {:2.3e}".format(global_lr))
                    print("weight_norm: {:2.3e}".format(weight_norm))
                    print("grad_norm: {:2.3e}".format(grad_norm))
                    print(torch.norm(d_p.add_(1e-16)))
                    print(self.prev_lr)
                    exit()

                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
                buf = param_state['momentum_buffer']
                to_add = d_p + weight_decay * p.data
                buf.mul_(momentum).add_(to_add,alpha=actual_lr)
                #buf.mul_(momentum).add_(actual_lr,d_p + weight_decay * p.data)
                p.data.add_(-buf)
                self.curr_lr.append(actual_lr)
        self.prev_lr = self.curr_lr

        return loss
