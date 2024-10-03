import torch
import torch.nn as nn
import torch.nn.functional as F
class TV(nn.Module):
    """
    Total Variation convex regularizer with learnable weights and smoothed L1 potential.
    """
    def __init__(self, in_channels, learnable_smoothing=False, make_non_learnable=False):
        super(TV, self).__init__()
        self.in_channels = in_channels  
        self.weights = nn.Parameter(-6.0 * torch.ones(in_channels))
        self.smoothing = None
        self.coef = nn.Parameter(torch.tensor(0.0))

        # Optional learnable smoothing
        if learnable_smoothing:
            self.smoothing = nn.Parameter(torch.tensor(-8.0))
        
        # Optionally make all existing parameters non-learnable
        if make_non_learnable:
            self.weights.requires_grad_(False)
            if self.smoothing is not None:
                self.smoothing.requires_grad_(False)
            self.coef.requires_grad_(False)  # Ensure this is also non-learnable if requested

    def forward_difference(self, x):
        # Compute finite differences in x and y directions
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    def smoothed_l1(self, x):
        # Smoothed L1 norm without learnable smoothing
        nabla_xy = torch.sqrt(x**2 + 1e-4) - 1e-8
        nabla_xy *= torch.exp(self.weights).view(1, -1, 1, 1)  # Adjusted view to generalize
        return torch.sum(nabla_xy)

    def smoothed_l1_learnable(self, x):
        # Smoothed L1 norm with learnable smoothing parameter
        smoothing_val = torch.exp(self.smoothing)
        nabla_xy = torch.sqrt(x**2 + smoothing_val) - smoothing_val
        nabla_xy *= torch.exp(self.weights).view(1, self.in_channels, 1, 1)  # Adjusted view to generalize
        return torch.sum(nabla_xy)

    def forward(self, x):
        # Compute finite differences
        dx, dy = self.forward_difference(x)
        
        # Pad dx and dy to ensure they have the same spatial dimensions
        dx_padded = F.pad(dx, (0, 1, 0, 0))  # Pad last dimension (width) by 1 on the right
        dy_padded = F.pad(dy, (0, 0, 0, 1))  # Pad second-to-last dimension (height) by 1 on the bottom
        
        # Combine the padded differences
        combined_diff = torch.cat([dx_padded, dy_padded], dim=-1)  # Concatenate along the last dimension

        # Use learnable smoothing if it's enabled
        if self.smoothing is not None:
            smoothed = self.smoothed_l1_learnable(combined_diff)
        else:
            smoothed = self.smoothed_l1(combined_diff)

        # Apply the learned coefficient scaling
        return torch.exp(self.coef) * smoothed
