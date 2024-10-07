import torch
import torch.nn as nn

class FoE(nn.Module):
    """
    Field of Experts convex regularizer with learnable weights and smoothed L1 potential.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, \
        learnable_smoothing = False, learnable_weights = False, zero_mean = False, make_non_learnable = False):
        super(FoE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, self.padding, bias=False, padding_mode='reflect')
        self.conv.weight.requires_grad = True
        self.weights = 0.01 * torch.ones(out_channels)
        # self.coef = nn.Parameter(torch.tensor(0.))
        self.smoothing = None
        self.learnable_weights = learnable_weights
        self.zm = zero_mean
        if learnable_smoothing:
            self.smoothing = nn.Parameter(torch.tensor(-3.0))
        if learnable_weights:
            self.weights = nn.Parameter(self.weights)
        self.coef = nn.Parameter(torch.tensor(0.))
           # Option to make everything non-learnable
        if make_non_learnable:
            self.make_all_non_learnable()

    def make_all_non_learnable(self):
        """Sets all parameters to non-learnable."""
        # Make convolution weights non-learnable
        self.conv.weight.requires_grad_(False)

        # Make smoothing non-learnable if it exists
        if self.smoothing is not None:
            self.smoothing.requires_grad_(False)

        # Make weights non-learnable
        if isinstance(self.weights, nn.Parameter):
            self.weights.requires_grad_(False)

        # Make coef non-learnable
        self.coef.requires_grad_(False)
    def init_weights(self):
        # zero-mean Xavier initialization
        weights = nn.init.xavier_normal_(self.conv.weight.data)
        self.conv.weight.data = weights - torch.mean(weights, dim=(2, 3), keepdim=True)
    def zero_mean(self):
        # Project the weights to have zero mean
        with torch.no_grad():
            self.conv.weight -= torch.mean(self.conv.weight, dim=(2, 3), keepdim=True)
    def smoothed_l1(self, x):
        # Smoothed L1 potential
        return torch.sum(torch.sqrt(x**2 + 1e-6) - 1e-3)
    def smoothed_l1_learnable(self, x):
        # Smoothed L1 potential with learnable smoothing parameter
        return torch.sum(torch.sqrt(x**2 + torch.exp(self.smoothing)) - torch.exp(self.smoothing))
    def forward(self, x):
        conv_output = self.conv(x)
        if self.zm:
            self.zero_mean()

        if self.learnable_weights:
            conv_output *= self.weights.view(-1, 1, 1)

        if self.smoothing is not None:
            return torch.exp(self.coef) * self.smoothed_l1_learnable(conv_output)
        else:
            return torch.exp(self.coef) * self.smoothed_l1(conv_output)