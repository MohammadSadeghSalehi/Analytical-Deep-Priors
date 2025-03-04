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

# load image
size_x = 300
size_y = 400
img_index = 15
channels = 3
noise_level = 5/255

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

problem_type = "deblurring" # "deblurring", "both_B" , "semiblind"
class Lower_Level(nn.Module):
    def __init__(self, measurement, regularizer, forward_operator = lambda x: x):
        super(Lower_Level, self).__init__()
        self.regularizer = regularizer
        self.measurement = measurement
        self.A = forward_operator
        self.param = 100
    def data_fidelity(self, x):
        # return 0.5 *torch.mean(torch.norm(self.A(x) - self.measurement, dim = (2, 3))**2)
        return 0.5 * torch.norm(self.A(x) - self.measurement)**2/ (self.measurement.shape[0])
        # return torch.nn.MSELoss()(self.A(x), self.measurement)
    def forward(self, x):
        return self.data_fidelity(x) + self.param * self.regularizer(x)
class Upper_Level(nn.Module):
    def __init__(self, y, operator, kernel, kernel_original):
        super(Upper_Level, self).__init__()
        self.y = y
        self.A = operator
        self.kernel = kernel   
        self.kernel_original = kernel_original.to(device)
        self.reg_param = 1
    def sobolev_norm(self):
        """
        Computes the Sobolev norm of the convolution operator defined by the kernel.
        """
        if self.kernel.device != self.kernel_original.device:
            self.kernel = self.kernel.to(self.kernel_original.device)
        self.kernel.requires_grad = True
        kernel = self.kernel - self.kernel_original
        # Compute L2 norm of the kernel (||k||_L2)
        l2_norm = torch.norm(kernel, p=2)

        # Compute gradients of the kernel in x and y directions (finite differences)
        grad_x = kernel[:, :, :, 1:] - kernel[:, :, :, :-1]  # Difference in x-direction
        grad_y = kernel[:, :, 1:, :] - kernel[:, :, :-1, :]  # Difference in y-direction

        # Compute L2 norm of the gradients (||grad_x||_L2 and ||grad_y||_L2)
        grad_x_norm = torch.norm(grad_x, p=2)
        grad_y_norm = torch.norm(grad_y, p=2)

        # Sobolev norm: ||A||_{H^1}^2 = ||k||_L2^2 + ||grad_x||_L2^2 + ||grad_y||_L2^2
        sobolev_norm = torch.sqrt(l2_norm**2 + grad_x_norm**2 + grad_y_norm**2)
        
        # Calculate gradients with respect to the kernel
        # Note: We need to set create_graph=True to compute higher-order derivatives
        grad_l2 = torch.autograd.grad(l2_norm, self.kernel, create_graph=True)[0]
        grad_grad_x = torch.autograd.grad(grad_x_norm, self.kernel, create_graph=True)[0]
        grad_grad_y = torch.autograd.grad(grad_y_norm, self.kernel, create_graph=True)[0]

        # Combine the gradients: we need to take the gradient of each component norm
        gradient_sobolev_norm = self.reg_param * (grad_l2 + grad_grad_x + grad_grad_y)
        return sobolev_norm, gradient_sobolev_norm
    def forward(self, x):
        # return  torch.norm(x - self.x)**2/ (self.x.shape[0] * self.x.shape[1])
        if problem_type == "both_B":
            B_conv = blur(self.kernel)
            return torch.norm(B_conv(x) - self.y)**2/ (self.y.shape[0] * self.y.shape[1]) + self.reg_param * self.sobolev_norm()[0]
        return torch.norm(self.A(x) - self.y)**2/ (self.y.shape[0] * self.y.shape[1]) + self.reg_param * self.sobolev_norm()[0]
def initialize_optimizer(hypergrad, alpha, optimizer_type='MAID'):
    hypergrad.warm_start = False
    if optimizer_type == 'MAID':
        hypergrad.verbose = False
        # hypergrad.warm_start = False
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
    logs_dict["lower_iter"].append(hypergrad.logs["lower_counter"])
    logs_dict["cg_iter"].append(hypergrad.logs["cg_counter"])
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

# kernel_4d = gaussian_kernel(5, 2, 1, channels)

class blur (nn.Module):
    def __init__(self, kernel):
        super(blur, self).__init__()
        self.kernel = kernel
        kernel_size = kernel.shape[-1]
        padding = kernel_size // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding = padding, bias = False, groups= channels, padding_mode= 'replicate')
        self.conv.weight.requires_grad = True
        self.conv.weight.data = kernel
    def forward(self, x):
        if x.device != self.conv.weight.data.device:
            self.conv.weight.data = self.conv.weight.data.to(x.device)
        # self.conv.weight.data = self.kernel
        return self.conv(x)

init_kernel2 = gaussian_kernel(5, 2, 0.5, channels) 
init_kernel = motion_blur_kernel(5, 'diagonal', channels)
# init_kernel = gaussian_kernel(5, 2, 2, channels)
# init_kernel = disc_blur_kernel(5, channels)

init_operator = blur(init_kernel)
init_operator_2 = blur(init_kernel2)
# plot kernel
plot_and_save_kernel(init_kernel, channels=channels, kernel_name="kernel_init", dpi=300)
kernel_4d = init_kernel.clone()
operator = blur(kernel_4d)
if problem_type == "semiblind":
    plot_and_save_kernel(init_kernel2, channels=channels, kernel_name="kernel_init2", dpi=300)
    img , noisy_img = load_image_and_add_noise(img_index, size_x, size_y, channels, init_operator, 0)
    noisy_img = init_operator_2(noisy_img) + torch.randn_like(noisy_img) * noise_level
else:
    img , noisy_img = load_image_and_add_noise(img_index, size_x, size_y, channels, operator, noise_level)
# visualise and save blurred image and original image
if channels == 1:
    plt.imshow(img.cpu().detach().squeeze().numpy(), cmap='gray')
else:
    plt.imshow(img.cpu().detach().squeeze().permute(1, 2, 0).numpy())
plt.axis('off')
plt.savefig(f'{os.getcwd()}/logs/original.png', bbox_inches='tight', dpi = 300)
plt.close()
if channels == 1:
    plt.imshow(noisy_img.cpu().detach().squeeze().numpy(), cmap='gray')
else:
    plt.imshow(noisy_img.cpu().detach().squeeze().permute(1, 2, 0).numpy())
plt.title('PSNR: {:.2f}'.format(psnr(img, noisy_img).mean().item()))
plt.axis('off')
plt.savefig(f'{os.getcwd()}/logs/noisy.png', bbox_inches='tight', dpi = 300)
plt.close()

init = noisy_img.clone()
# init = torch.zeros_like(init)
# lower level regularizer
regularizer = TV(channels, learnable_smoothing=False , make_non_learnable=True).to(device)
# regularizer = FoE(channels,10, 7,  learnable_smoothing = True, learnable_weights = True, make_non_learnable = False).to(device)
# regularizer.load_state_dict(torch.load(f'{os.getcwd()}/regularizer.pt'))
# regularizer.init_weights()
regularizer.forward_operator = operator
# ADP lower level and upper level
Lower_Level_obj = Lower_Level(noisy_img, regularizer)
lower_level = Lower_Level(noisy_img.to(device), regularizer, forward_operator= operator)#.to(device)
upper_level = Upper_Level(noisy_img.to(device), operator, kernel_4d, kernel_4d).to(device)
x_init = init.to(device)
hypergrad = Hypergrad_Calculator(lower_level, upper_level, x_init = x_init, verbose= True)

# regularized solution
classic_solution = hypergrad.FISTA(x_init, 1e-3, 100)
if channels == 1:
    plt.imshow(classic_solution.cpu().detach().squeeze().numpy(), cmap='gray')
else:
    plt.imshow(classic_solution[:,:,:,:].cpu().detach().squeeze().permute(1, 2, 0).numpy(), vmin=0, vmax=1)
plt.axis('off')
plt.title('PSNR: {:.2f}'.format(psnr(img, classic_solution.detach().cpu()).mean().item()))
plt.savefig(f'{os.getcwd()}/logs/classic_solution.png', bbox_inches='tight', dpi = 300)
# plt.show()
plt.close()
def main_solver(hypergrad, data, noisy, init, device, psnr_fn, upper_iter, alpha, eps0, setting , mode, budget, threshold):
    optimizer = initialize_optimizer(hypergrad, alpha, optimizer_type='MAID')
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
        
        
        # Kernel Projection
        kernel = hypergrad.lower_level_obj.regularizer.forward_operator.conv.weight.data
        # with torch.no_grad():
        #     kernel/= torch.sum(torch.abs(kernel))
        hypergrad.lower_level_obj.regularizer.forward_operator.conv.weight.data = kernel
        hypergrad.upper_level_obj.kernel = kernel
        hypergrad.lower_level_obj.A.conv.weight.data = kernel
        
        
        # grad += torch.norm(torch.cat([p.grad.flatten() for p in optimizer.param_groups[0]['params']]))
        hypergrad.x_init = init
        hypergrad.lower_level_obj.regularizer.load_state_dict(optimizer.hypergrad.lower_level_obj.regularizer.state_dict())
        # Log PSNR and loss
        psnr_value = psnr_fn(data, init.cpu()).mean().item()
        print("PSNR: ", psnr_value, "Loss: ", loss_val.item())
        if channels == 1:
            plt.imshow(init.cpu().detach().squeeze().numpy(), cmap='gray')
        else:
            plt.imshow(init.cpu().detach().squeeze().permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f'PSNR: {psnr_value:.2f}')
        plt.savefig(f'{directory}/logs/ADP.png', bbox_inches='tight', dpi=300)
        plt.close()
            
        # plotting kernel_differences
        kernel = hypergrad.lower_level_obj.regularizer.forward_operator.conv.weight.data.to(kernel_4d.device)
        kernel_diff = torch.abs(kernel - kernel_4d).squeeze().cpu().detach().numpy()
        plot_and_save_kernel(kernel_diff, channels=channels, kernel_name="kernel_diff", dpi=300)
        #plotting kernel
        plot_and_save_kernel(kernel, channels=channels, kernel_name="kernel", dpi=300)
        loss.append(loss_val.item())
        psnr.append(psnr_value)
        
        # logging
        log_metrics(logs_dict, hypergrad, optimizer, data, psnr_fn, directory, eps0, alpha, setting)
        if hypergrad.lower_tol < threshold:
            print("Stopping Criterion")
            break
           
    # Save the final logs
    torch.save(logs_dict, f'{directory}/Logs/final_logs_dict.pt')
    
# Run the main solver
upper_iter = 41
alpha = 1e-6
eps0 = 1e0
setting = "motion_blur"
mode = problem_type + "TV_semi_blind"  #"TV_fixed" "FoE_learnable" "TV_semi_blind" 
budget = 10
threshold = 1e-6
hypergrad.upper_level_obj.reg_param = 100
hypergrad.lower_level_obj.param = 100
hypergrad.lower_tol = eps0
logsdict = initialize_logs(setting, mode)
logsdict = initialize_log_values(img, noisy_img, init, hypergrad, logsdict, device, eps0, alpha, psnr)
main_solver(hypergrad, img, noisy_img, init, device, psnr, upper_iter, alpha, eps0, setting, mode, budget, threshold)