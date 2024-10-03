import torch
import torch.nn as nn

class TV(nn.Module):
    """""
    Total Variation convex regularizer with learnable weights and smoothed L1 potential.
    """""
    def __init__(self, in_channels, learnable_smoothing = False):
        super(TV, self).__init__()
        self.in_channels = in_channels  
        self.weights = nn.Parameter(-6.0 *torch.ones(in_channels))
        self.smoothing = None
        if learnable_smoothing:
            self.smoothing = nn.Parameter(torch.tensor(-3.0))
    def forward_differece(self, x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy
    def smoothed_l1(self, x):
        # Smoothed L1
        nabla_xy = torch.sqrt(x**2 + 1e-6) - 1e-3
        nabla_xy *= torch.exp(self.weights).view( 1, -1, 1)
        return torch.sum(nabla_xy)
    def smoothed_l1_learnable(self, x):
        # Smoothed L1 with learnable smoothing parameter
        nabla_xy = torch.sqrt(x**2 + torch.exp(self.smoothing)) - torch.exp(self.smoothing)
        nabla_xy *= torch.exp(self.weights).view(1, self.in_channels, 1)
        return torch.sum(nabla_xy)
    def forward(self, x):
        dx, dy = self.forward_differece(x)
        if self.smoothing is not None:
            return self.smoothed_l1_learnable(dx.view(2, 3, -1) + dy.view(2, 3, -1))
        else:
            return self.smoothed_l1(dx.view(2, 3, -1) + dy.view(2, 3, -1))
    