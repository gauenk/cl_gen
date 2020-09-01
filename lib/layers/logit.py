import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, simCLR, n_classes):
        super(LogisticRegression, self).__init__()
        self.simCLR = simCLR
        n_features = simCLR.n_features
        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        with torch.no_grad():
            h,_,z,_ =self.simCLR(x,x)
        return self.model(h)
