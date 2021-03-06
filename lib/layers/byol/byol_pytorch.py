import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T

# import contrastive learning stuff
from easydict import EasyDict as edict
from layers.simcl import ClBlockLoss

# -- transforms for _supervised_ augmentations --
from datasets.transforms.noise import AddGaussianNoise,AddPoissonNoiseBW


# helper functions

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class PickOnlyOne(nn.Module):
    def __init__(self, fn_l):
        super().__init__()
        self.fn_l = fn_l

    def __call__(self):
        fn = random.choice(self.fn_l)
        return fn

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_embedding = False):
        representation = self.get_representation(x)

        if return_embedding:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation

# main class

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        batch_size = 2,
        rand_batch_size = 2,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True,
        patch_helper = None,
    ):
        super().__init__()
        self.net = net
        self.patch_helper = patch_helper

        # default SimCLR augmentation

        """
        In our function, color is an important property
        """
        DEFAULT_AUG = torch.nn.Sequential(
            # RandomApply(
            #     T.ColorJitter(0.8, 0.8, 0.8, 0.2),
            #     p = 0.3
            # ),
            # T.RandomGrayscale(p=0.5),
            # T.RandomHorizontalFlip(),
            # RandomApply(
            #     T.GaussianBlur((3, 3), (1.0, 2.0)),
            #     p = 0.2
            # ),
            # T.RandomResizedCrop((image_size, image_size)),
            # T.Normalize(
            #     mean=torch.tensor([0.485, 0.456, 0.406]),
            #     std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        def null(image): return image
        gn,pn = AddGaussianNoise(std=75.),AddPoissonNoiseBW(4.),
        self.gn,self.pn = gn,pn
        self.choose_noise = PickOnlyOne([gn,null])

        self.online_encoder = NetWrapper(net, projection_size,
                                         projection_hidden_size, layer=hidden_layer)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # simclr since byol isn't working well
        hyper = edict()
        hyper.temperature = 1
        self.simclr_loss = ClBlockLoss(hyper,2,batch_size)


        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        rdata = torch.abs(torch.randn(rand_batch_size, 3, image_size, image_size, device=device))
        self.forward(rdata)

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, x, return_embedding = False):
        
        if return_embedding:
            return self.online_encoder(x,True)


        # -- this works for Gaussian(75); basically just denoising --
        # PSNR: 22.74 v.s. 27.46 & 22.36 v.s. 26.85 | Harms clean images; 29 v 26 and 31 v 42
        noisy_xform = self.choose_noise()
        order = torch.randperm(2)
        # noisy1,noisy2 = noisy_xform(x),noisy_xform(x)
        noisy1,noisy2 = self.gn(x),x
        noisy_l = [noisy1,noisy2]
        noisy1,noisy2 = noisy_l[order[0]],noisy_l[order[1]]

        # -- test 2 --
        # noisy_xform1,noisy_xform2 = self.choose_noise(),self.choose_noise()
        # noisy1,noisy2 = noisy_xform1(x),noisy_xform2(x)

        image_one, image_two = self.augment1(noisy1), self.augment2(noisy2)

        online_proj_one, _ = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        simclr_input = torch.stack([online_pred_one,online_pred_two],dim=0)
        loss_simclr = self.simclr_loss(simclr_input)

        loss = 0.1*(loss_one + loss_two) + loss_simclr
        return loss.mean()
