from hg import *
from loader import *
import torch
import sys
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *
from MAID import *
from TV import *
from FoE import *

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
class Lower_Level(nn.Module):
    def __init__(self, measurement, regularizer, forward_operator = lambda x: x):
        super(Lower_Level, self).__init__()
        self.regularizer = regularizer
        self.measurement = measurement
        self.A = forward_operator
        self.param = 2.5
    def data_fidelity(self, x):
        # return 0.5 *torch.mean(torch.norm(self.A(x) - self.measurement, dim = (2, 3))**2)
        return 0.5 * torch.norm(self.A(x) - self.measurement)**2/ (self.measurement.shape[0])
        # return torch.nn.MSELoss()(self.A(x), self.measurement)
    def forward(self, x):
        return self.data_fidelity(x) + self.param * self.regularizer(x)
class Upper_Level(nn.Module):
    def __init__(self, y, operator, kernel):
        super(Upper_Level, self).__init__()
        self.y = y
        self.A = operator
        self.kernel = kernel   
    def forward(self, x):
        # return  torch.norm(x - self.x)**2/ (self.x.shape[0] * self.x.shape[1])
        return torch.norm(self.A(x) - self.y)**2/ (self.y.shape[0] * self.y.shape[1])
def initialize_optimizer(hypergrad, alpha, optimizer_type='MAID'):
    if optimizer_type == 'MAID':
        hypergrad.verbose = False
        hypergrad.warm_start = False
        return MAID(hypergrad.lower_level_obj.regularizer.parameters(), lr=alpha, hypergrad_module=hypergrad, eps=eps0)
    # Fixed accuracy and step size GD
    return MAID(hypergrad.lower_level_obj.regularizer.parameters(), lr=alpha, hypergrad_module=hypergrad, eps=eps0, fixed_eps=True, fixed_lr=True)
def initialize_logs(setting, mode):
    return {
        "loss": [], "eps": [], "step": [],
        "lower_iter": [], "cg_iter": [], "psnr": [],
        "setting": setting, "mode": mode
    }
def initialize_log_values(data, noisy, init, hypergrad, logs_dict, device, eps0, alpha, psnr_fn):
    # Initialize logs 
    hypergrad = data_update(hypergrad, init.to(device), noisy.to(device), data.to(device))
    logs_dict["loss"].append(hypergrad.upper_level_obj(hypergrad.x_init).item())
    logs_dict["psnr"].append(psnr_fn(data, init.cpu()).mean().item())
    logs_dict["eps"].append(eps0)
    logs_dict["step"].append(alpha)
    logs_dict["lower_iter"].append(hypergrad.logs["lower_counter"])
    logs_dict["cg_iter"].append(hypergrad.logs["cg_counter"])
    return logs_dict
def update_data(hypergrad, init_data, noisy_data, data, device):
    return data_update(hypergrad, init_data.to(device), noisy_data.to(device), data.to(device))

def log_metrics(logs_dict, hypergrad, optimizer, data, psnr_fn, directory, eps0, alpha, setting):
    logs_dict["eps"].append(hypergrad.lower_tol)
    logs_dict["step"].append(optimizer.lr)
    logs_dict["psnr"].append(psnr_fn(data, hypergrad.x_init.cpu()).mean().item())
    save_logs(logs_dict, hypergrad, directory, eps0, alpha, setting)

def save_logs(logs_dict, hypergrad, directory, eps0, alpha, setting):
    # Define file paths based on the setting
    log_file = f'{directory}/Logs/logs_dict_{eps0}_{alpha}_{setting}.pt'
    
    if setting == "constant":
        regularizer_file = f'{directory}/logs/regularizer_{eps0}_{alpha}_{setting}.pt'
    else:
        regularizer_file = f'{directory}/logs/regularizer_{eps0}_{alpha}_{setting}.pt'
    # Save logs and regularizer state
    torch.save(logs_dict, log_file)
    torch.save(hypergrad.lower_level_obj.regularizer.state_dict(), regularizer_file)

# load image
size_x = 270
size_y = 270
img_index = 4
channels = 3
noise_level = 0.01

blur_kernel_original = gaussian_kernel(5, 2, 1)
kernel_4d = blur_kernel_original.unsqueeze(0).unsqueeze(0)
kernel_4d = kernel_4d.repeat(3, 1, 1, 1)
blur_original = nn.Conv2d(channels, channels, 5, padding = 2, bias = False, groups= channels)
blur_original.weight.data = kernel_4d
class blur (nn.Module):
    def __init__(self, kernel):
        super(blur, self).__init__()
        self.kernel = kernel
        self.conv = nn.Conv2d(channels, channels, 5, padding = 2, bias = False, groups= channels, padding_mode= 'reflect')
        self.conv.weight.requires_grad = True
    def forward(self, x):
        if x.device != self.kernel.device:
            self.kernel = self.kernel.to(x.device)
            self.conv = self.conv.to(x.device)
        self.conv.weight.data = self.kernel
        return self.conv(x)
blur_obj = blur(kernel_4d)



operator = blur_obj

img , noisy_img = load_image_and_add_noise(img_index, size_x, size_y, channels, operator, noise_level)
init = noisy_img.clone()
init = torch.zeros_like(init)
# lower level regularizer
regularizer = TV(channels, learnable_smoothing=False, make_non_learnable=True).to(device)
regularizer.forward_operator = operator
# ADP lower level and upper level
Lower_Level_obj = Lower_Level(noisy_img, regularizer)
lower_level = Lower_Level(noisy_img, regularizer, forward_operator= operator)#.to(device)
upper_level = Upper_Level(noisy_img.to(device), operator, blur_kernel_original).to(device)
x_init = init.to(device)
hypergrad = Hypergrad_Calculator(lower_level, upper_level, x_init = x_init, verbose= True)

def main_solver(hypergrad, data, noisy, init, device, psnr_fn, upper_iter, alpha, eps0, setting , mode, budget, threshold):
    optimizer = initialize_optimizer(hypergrad, alpha)
    logs_dict = initialize_logs(setting, mode)
    logs_dict = initialize_log_values(data, noisy, init, hypergrad, logs_dict, device, eps0, alpha, psnr_fn)
    directory = os.getcwd()
    progress_bar = tqdm(range(upper_iter), desc='Upper iter: ')
    loss, psnr = [], []
    directory = os.getcwd()
    grad = 0
    for i in progress_bar:
        # Update hypergrad with new data
        hypergrad = update_data(hypergrad, init, noisy, data, device)
        optimizer.hypergrad = hypergrad

        optimizer.zero_grad()
        optimizer.hypergrad.hypergrad()
        hypergrad.lower_skip = True

        # Perform optimization step
        loss_val, init = optimizer.step()
        # grad += torch.norm(torch.cat([p.grad.flatten() for p in optimizer.param_groups[0]['params']]))
        hypergrad.x_init = init
        hypergrad.lower_level_obj.regularizer.load_state_dict(optimizer.hypergrad.lower_level_obj.regularizer.state_dict())
        # Log PSNR and loss
        psnr_value = psnr_fn(data, init.cpu()).mean().item()
        print("PSNR: ", psnr_value, "Loss: ", loss_val.item())
        if i % 10 == 0:
            plt.imshow(init.cpu().detach().squeeze().permute(1, 2, 0).numpy())
            plt.show()
            plt.savefig(f'{directory}/logs/ADP.png')
            kernel = hypergrad.lower_level_obj.regularizer.forward_operator.conv.weight.data.squeeze().cpu().detach().numpy()
            plt.imshow(kernel[0])
            plt.colorbar()
            plt.show()
            plt.savefig(f'{directory}/logs/ADP_kernel.png')
            print("kernel difference: ", torch.norm(torch.tensor(kernel) - kernel_4d.cpu().detach().squeeze()).item())
        loss.append(loss_val.item())
        psnr.append(psnr_value)
        if hypergrad.lower_tol < threshold:
            print("Stopping Criterion")
            break
        
    
    # Save the final logs
    torch.save(logs_dict, f'{directory}/Logs/final_logs_dict.pt')
    
# Run the main solver
upper_iter = 100
alpha = 1e-6
eps0 = 1
setting = "Gaussian_blur"
mode = "TV_fixed"
budget = 10
threshold = 1e-3
hypergrad.lower_tol = eps0
main_solver(hypergrad, img, noisy_img, init, device, psnr, upper_iter, alpha, eps0, setting, mode, budget, threshold)