from hg import *
from optimizer import *
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


class Lower_Level(nn.Module):
    def __init__(self, measurement, regularizer, forward_operator = lambda x: x):
        super(Lower_Level, self).__init__()
        self.regularizer = regularizer
        self.measurement = measurement
        self.A = forward_operator
    def data_fidelity(self, x):
        # return 0.5 *torch.mean(torch.norm(self.A(x) - self.measurement, dim = (2, 3))**2)
        return 0.5 * torch.norm(self.A(x) - self.measurement)**2/ (self.measurement.shape[0])
        # return torch.nn.MSELoss()(self.A(x), self.measurement)
    def forward(self, x):
        return self.data_fidelity(x) + self.regularizer(x)
class Upper_Level(nn.Module):
    def __init__(self, x):
        super(Upper_Level, self).__init__()
        self.x = x
    def forward(self, x):
        return  torch.norm(x - self.x)**2/ (self.x.shape[0] * self.x.shape[1])
    
def initialize_optimizer(hypergrad, alpha, optimizer_type='MAID'):
    if optimizer_type == 'MAID':
        hypergrad.verbose = False
        hypergrad.warm_start = False
        return MAID(hypergrad.lower_level_obj.regularizer.parameters(), lr=alpha, hypergrad_module=hypergrad, eps=eps0)
    # Fixed accuracy and step size GD
    return MAID(hypergrad.lower_level_obj.regularizer.parameters(), lr=alpha, hypergrad_module=hypergrad, eps=eps0, fixed_eps=True, fixed_lr=True)
def initialize_logs(setting, p, q, training_mode):
    return {
        "loss": [], "eps": [], "step": [],
        "lower_iter": [], "cg_iter": [], "psnr": [],
        "setting": setting, "p": p, "q": q, "train_mode": training_mode
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