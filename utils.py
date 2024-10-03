import torch
import deepinv as dinv
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

def data_update(hypergrad, lower_init, lower_data, upper_data):
    hypergrad.lower_level_obj.measurement = lower_data
    hypergrad.upper_level_obj.x = upper_data
    hypergrad.x_init = lower_init
    return hypergrad
def total_iter(batch, epoch, n_batches):
    return batch + epoch * n_batches