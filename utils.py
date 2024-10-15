import torch
import deepinv as dinv
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize, SymLogNorm
import matplotlib.pyplot as plt
import os
import numpy as np
# def psnr(img1, img2):
#     img1 = torch.clip(img1, 0, 1)
#     img2 = torch.clip(img2, 0, 1)
#     mse = torch.mean((img1 - img2) ** 2)
#     return 20 * torch.log10(1.0 / torch.sqrt(mse))

psnr = dinv.loss.PSNR(max_pixel=1, normalize=False)
# def gaussian_kernel(size: int, mean: float, std: float):
#     """Creates a 2D Gaussian Kernel for convolution."""
#     d = torch.arange(size).float() - mean
#     gaussian_1d = torch.exp(-(d ** 2) / (2 * std ** 2))
#     gaussian_1d = gaussian_1d / gaussian_1d.sum()
    
#     # Create a 2D Gaussian kernel by computing the outer product of two 1D kernels
#     gaussian_2d = torch.outer(gaussian_1d, gaussian_1d)
    
#     # Normalize the 2D kernel to ensure the sum equals 1
#     gaussian_2d = gaussian_2d / gaussian_2d.sum()
    
#     return gaussian_2d

def gaussian_kernel(size: int, mean: float, std: float, channels: int):
    """Creates a 4D Gaussian Kernel for convolution, handling grayscale or colored images."""
    
    # Create 1D Gaussian kernel
    d = torch.arange(size).float() - mean
    gaussian_1d = torch.exp(-(d ** 2) / (2 * std ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()  # Normalize to sum to 1
    
    # Create a 2D Gaussian kernel by outer product of the 1D kernels
    gaussian_2d = torch.outer(gaussian_1d, gaussian_1d)
    gaussian_2d = gaussian_2d / gaussian_2d.sum()  # Normalize again
    
    # Add dimensions to make it 4D (needed for convolution)
    kernel_4d = gaussian_2d.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, size, size)
    # Repeat the kernel for the number of channels
    kernel_4d = kernel_4d.repeat(channels, 1, 1, 1)  # Shape: (channels, 1, size, size)
    
    return kernel_4d

# Motion blur kernel: Vertical, Horizontal, Diagonal
def motion_blur_kernel(size: int, blur_type: str, channels: int):
    """
    Creates a 4D Motion Blur Kernel for convolution.
    blur_type can be 'vertical', 'horizontal', or 'diagonal'.
    """
    kernel_2d = torch.zeros((size, size))
    
    if blur_type == 'vertical':
        kernel_2d[:, size // 2] = 1.0  # Middle column is 1
    elif blur_type == 'horizontal':
        kernel_2d[size // 2, :] = 1.0  # Middle row is 1
    elif blur_type == 'diagonal':
        for i in range(size):
            kernel_2d[i, i] = 1.0  # Diagonal elements are 1

    # Normalize kernel to sum to 1
    kernel_2d = kernel_2d / kernel_2d.sum()

    # Expand to 4D and repeat for the number of channels
    kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, size, size)
    kernel_4d = kernel_4d.repeat(channels, 1, 1, 1)  # Repeat for the number of channels
    
    return kernel_4d

# Disc blur kernel
def disc_blur_kernel(size: int, channels: int):
    """
    Creates a 4D Disc Blur Kernel for convolution.
    The kernel is a circular mask with a smooth edge.
    """
    radius = size // 2
    y, x = torch.meshgrid(torch.arange(-radius, radius + 1), torch.arange(-radius, radius + 1))
    distance = torch.sqrt(x ** 2 + y ** 2)
    
    # Create a disc mask where distance is less than or equal to the radius
    disc_2d = (distance <= radius).float()
    
    # Normalize kernel to sum to 1
    disc_2d = disc_2d / disc_2d.sum()

    # Expand to 4D and repeat for the number of channels
    kernel_4d = disc_2d.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, size, size)
    kernel_4d = kernel_4d.repeat(channels, 1, 1, 1)  # Repeat for the number of channels
    
    return kernel_4d

def data_update(hypergrad, lower_init, lower_data, upper_data):
    hypergrad.lower_level_obj.measurement = lower_data
    hypergrad.upper_level_obj.x = upper_data
    hypergrad.x_init = lower_init
    return hypergrad
def total_iter(batch, epoch, n_batches):
    return batch + epoch * n_batches

def plot_and_save_kernel(kernel, channels=1, save_path=None, kernel_name="kernel", dpi=300):
    """
    Plots and saves kernel images for 1-channel or multi-channel data with SymLogNorm scaling.
    
    Args:
        kernel (torch.Tensor): The kernel tensor to plot, expected to be on CPU.
        channels (int): Number of channels in the kernel (1 for grayscale, 3 for RGB).
        save_path (str, optional): Directory to save the output images. Defaults to current working directory.
        kernel_name (str, optional): Name prefix for the saved images. Defaults to 'kernel'.
        dpi (int, optional): DPI for the saved image. Defaults to 300.
    """
    # Set default save path if not provided
    if save_path is None:
        save_path = os.path.join(os.getcwd(), 'logs')
    # cmap = 'seismic'
    cmap = 'viridis'
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Get numpy representation of the kernel for normalization
    if type(kernel) == torch.Tensor:
        kernel_np = kernel.cpu().detach().numpy()
    else:
        kernel_np = kernel
    # Apply symmetric logarithmic normalization
    linthresh = 1e-6
    if np.min(kernel_np) > 0: # If all values are positive
        norm = LogNorm(vmin=np.min(kernel_np), vmax=np.max(kernel_np))
    else:
        norm = SymLogNorm(linthresh=linthresh, vmin=np.min(kernel_np), vmax=np.max(kernel_np), base=10)
    if kernel_name == "kernel_diff":
        norm = Normalize(vmin=0, vmax=np.max(kernel_np))
    # Plot for a single channel
    if channels == 1:
        plt.imshow(kernel_np.squeeze(), norm=norm, cmap=cmap)
        plt.colorbar()
        plt.savefig(f'{save_path}/{kernel_name}.png', bbox_inches='tight', dpi=dpi)
        plt.close()
    # Plot for three channels
    else:
        kernel = torch.tensor(kernel)
        for i in range(min(channels, 3)):  # Plot the first 3 channels if channels > 3
            plt.imshow(kernel[i].cpu().detach().squeeze().numpy(), norm=norm, cmap=cmap)
            cbar = plt.colorbar()
            # cbar.set_ticks([np.min(kernel_np), -1e-6, 0, 1e-6, np.max(kernel_np)])
            # cbar.set_ticklabels([f'{np.min(kernel_np):.1e}', '-1e-6', '0', '1e-6', f'{np.max(kernel_np):.1e}'])
            plt.savefig(f'{save_path}/{kernel_name}_{i+1}c.png', bbox_inches='tight', dpi=dpi)
            plt.close()