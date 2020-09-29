import torch,torchvision
import torch.nn as nn
import torch.nn.functional as F

class Projector(nn.Module):

    def __init__(self, n_features=128, projection_dim=64):
        super(Projector, self).__init__()

        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, projection_dim, bias=False),
        )

    def forward(self, x):
        return self.projector(x)

