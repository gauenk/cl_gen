
# python imports
from pathlib import Path

# pytorch imports
import torch
import torch.nn as nn
import torchvision

# project imports

# local imports
from .identity import Identity
from .resnet import get_resnet


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, args, encoder, n_features):
        super(SimCLR, self).__init__()

        self.normalize = args.normalize
        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, args.projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        if self.normalize:
            z_i = nn.functional.normalize(z_i, dim=1)
            z_j = nn.functional.normalize(z_j, dim=1)

        return h_i, h_j, z_i, z_j


def load_simclr(cfg):
    # initialize ResNet model
    encoder = get_resnet(cfg.cl.resnet, cfg.cl.dataset.name, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    model = SimCLR(cfg.simclr, encoder, n_features)
    if cfg.cl.load:
        fn = Path("checkpoint_{}.tar".format(cfg.cl.epoch_num))
        model_fp = cfg.cl.model_path / fn
        model.load_state_dict(torch.load(model_fp, map_location=cfg.cl.device.type))
    model = model.to(cfg.cl.device)

    # optimizer / loss
    cfg_adam = cfg.cl.optim.adam    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_adam.lr)  # TODO: LARS
    scheduler = None

    return model,optimizer,scheduler

