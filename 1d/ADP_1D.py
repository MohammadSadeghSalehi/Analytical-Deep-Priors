import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
import time as T

torch.manual_seed(0)
np.random.seed(0)

# Set tick font size and legend font size globally
plt.rcParams['xtick.labelsize'] = 24  # X-axis tick font size
plt.rcParams['ytick.labelsize'] = 24  # Y-axis tick font size
plt.rcParams['legend.fontsize'] = 26  # Legend font size

# Optional: Set font sizes for other elements for consistency
plt.rcParams['axes.titlesize'] = 26     # Title font size
plt.rcParams['axes.labelsize'] = 20     # Axis label font size
plt.rcParams['font.size'] = 20   

def psnr(signal1, signal2):
    signal1 = torch.clip(signal1, 0, 1)
    signal2 = torch.clip(signal2, 0, 1)
    mse = torch.mean((signal1 - signal2) ** 2)
    max = torch.max(signal1)
    return 20 * torch.log10(max / torch.sqrt(mse))

# Data: different functions on the real interval [-1,1]
# Operators: Integration operator and convolution with gaussian kernel

# discrete setting
h=0.01
t=torch.arange(-1+h/2,1,h,dtype=torch.float)

#############################################
#
# here are three ground truth examples:
#
#############################################

bsp = 0

if bsp == 0:
    x = 1*(t>-0.6) - 0.7*(t>-0.5) - 0.3*(t>0) + 0.7*(t>0.2) - 0.7*(t>0.5)
if bsp == 1:
    x = 0.8*torch.exp(-32*(t+0.3)**2) + 0.4*torch.exp(-16*(t-0.1)**2)
if bsp == 2:
    x = (t+0.7)*(t>-0.7) - (2*t + 1)*(t>-0.5) + (t + 0.3)*(t>-0.3) + (0.9-t)*(t>-0.1) \
        + (t-0.2)*(t>0.2) - 0.7*(t>0.4) + (3*t-1.8)*(t>0.6) - (3*t-1.8)*(t>0.8)

#############################################
#
# the forward oparator: integration=0, convolution=1
#
#############################################

forward = 1

A = torch.zeros((x.shape[0],x.shape[0]), dtype=torch.float)
if forward==0:
    for i in range(x.shape[0]):
        A[i,:i+1]=h
if forward==1:
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            A[i,j] = 1/(np.sqrt(2*np.pi)*5)*np.exp(-1/50*(i-j)**2)

#############################################
#
# create noisy data
#
#############################################
ground_truth = x
y = torch.matmul(A,x.reshape(x.shape[0],1))

eta = 0.005*torch.randn(y.shape, dtype=y.dtype)

ydelta = y + eta

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(t,x, label='truth')
plt.subplot(1,2,2)
plt.plot(t,y, label='data')
plt.plot(t,ydelta, label='noisy data')
plt.legend()
plt.savefig('data.png', dpi=300, bbox_inches='tight')
plt.close()

x_norm2 = (x**2).sum()*h
y_norm2 = (ydelta**2).sum()*h

print('Norm^2 of x: ' + str(x_norm2.numpy()))
print('Norm^2 of ydelta: ' + str(y_norm2.numpy()))

####################
#
# classical Tikhonov solution (for comparison and as initial value for further methods)
#
####################

#Tikhonov parameters for different examples
param_tik = [[0.0006, 0.003, 0.001],[0.0015,0.015,0.0012]]

alph = param_tik[forward][bsp]

M = torch.matmul(A.transpose(0,1),A) + alph*torch.eye(x.shape[0])
b = torch.matmul(A.transpose(0,1),ydelta)

x_tik = torch.linalg.solve(M,b)

print('R(x_tik) = ' + str((alph*(x_tik**2).sum()*h).numpy()))
print('reconstruction error of x_tik: ' + format(((x_tik.reshape(x.shape)-x)**2).sum()*h,'.3g'))



####################
#
# classical elastic-net solution (for comparison)
#
####################

def proxl1(x, alph):
    # shrinkage
    xn = F.relu(F.relu(x)-alph) - F.relu(F.relu(-x)-alph)
    return xn

# elastic-net parameters for different examples
param_eln = np.array([[[0.0012, 0.0002],[0.0015,0.0015],[0.0012,0.0004]],
                      [[0.004, 0.001],[0.002,0.004],[0.0018,0.0006]]])

alph_l1 = param_eln[forward][bsp][0]
alph_l2 = param_eln[forward][bsp][1]

lamb = 0.05
xk = x_tik
for k in range(10000):
    xkn=xk-lamb*(torch.matmul(A.transpose(0,1),torch.matmul(A,xk)-ydelta))-lamb*alph_l2*xk
    xkn = proxl1(xkn,alph_l1*lamb)
    if abs(xkn-xk).max()<0.000001:
        print(k)
        break
    xk=xkn

x_eln = xk

plt.figure(dpi=100)
plt.plot(t,x,label='truth')
plt.plot(t,x_tik, label='tik')
plt.plot(t,x_eln, label='eln')
plt.legend()
plt.savefig('tik_eln.png', dpi=300, bbox_inches='tight')
plt.close()


print('R(x_eln) = ' + str((alph_l1*abs(x_eln).sum()*h + alph_l2*(x_eln**2).sum()*h).numpy()))
print('reconstruction error of x_eln: ' + format(((x_eln.reshape(x.shape)-x)**2).sum()*h,'.3g'))

#############################
#
# DIP model with LISTA network for solving ADP
#
############################
time_unroll = []
time_unroll.append(0)
loss_unroll = []
unroll_iter = []
xk = torch.zeros_like(xk)
loss_unroll.append(0.5* torch.linalg.norm(torch.matmul(A,xk)-ydelta)**2)
unroll_iter.append(0)
time_start = T.time()

class ADP_eln(LightningModule):
    # The Analytic Deep Prior Model
    def __init__(self, A, y, x0, al_l1, al_l2, lamb, recursive, gr_tr, other=None, unroll_iter=[]):
        super(ADP_eln, self).__init__()
        
        self.register_buffer('A', A) # forward operator
        self.register_buffer('y', y) # noisy data
        self.al_l1 = al_l1 # l1-regularization parameter
        self.al_l2 = al_l2 # l2-regularization parameter
        self.lamb = lamb # (L)ISTA stepsize
        if x0 == None:
            self.register_buffer('x_iter', 0.5+0.25*torch.randn(y.shape, dtype=y.dtype)) # random initial value
        else:
            self.register_buffer('x_iter', x0) # given initial value

        self.B = torch.nn.parameter.Parameter(data=A.clone().detach()) # trainable linear operator
        #self.B = torch.nn.parameter.Parameter(data=-0.5*A.clone().detach()+1.5*torch.eye(y.shape[0], dtype=y.dtype))
        #self.B = torch.nn.parameter.Parameter(data=torch.eye(y.shape[0], dtype=y.dtype))
        
        self.results_adp = {}
        self.operators = {}
        self.k = 1
        self.unroll_iter = unroll_iter
        self.total_iter = 0
        self.recursive = recursive # wether 'input' = 'output of the last training step'
        
        self.gr_tr = gr_tr # ground truth for logging
        self.other = other # other reconstruction for comparison
        self.final = None
    def forward(self,x):
        for i in range(50):
            Bx = torch.matmul(self.B,x)
            x=x-self.lamb*(torch.matmul(self.B.transpose(0,1),Bx-self.y)+self.al_l2*x) #gradient step
            x = proxl1(x,self.al_l1*self.lamb) #prox step

        return x
    
    def training_step(self, batch, batch_idx):
        
        x=self.x_iter
        
        # forward pass
        x = self.forward(x)
        loss_unroll.append((torch.linalg.norm(torch.matmul(A,x)-ydelta)**2).detach().numpy())
        time_unroll.append(T.time()-time_start)
        self.total_iter += 50
        self.unroll_iter.append(self.total_iter)
        Ax_out = torch.matmul(self.A,x)
        
        # overwrite x_iter with output to increase the number of layers during training
        if self.recursive:
            self.x_iter = x.detach()
        
        # calculate the loss
        loss = F.mse_loss(Ax_out,self.y)
        
        self.log('train_loss', loss)
        
        # saving intermediate results
        if self.k in [1,2,3,5,10,20,30,50,75,100,200,300,500,750,1000,
                      1250,1500,1750,2000,2250,2500,2750,3000,3250,3500,3750,4000,5000,7500,10000]:
            if self.recursive:
                self.results_adp[self.k] = self.x_iter.detach()
            else:
                self.results_adp[self.k] = x.detach()
            self.operators[self.k] = self.B.detach()
        
        self.k = self.k+1
        self.final = x.detach()
        return loss
    
    def on_train_epoch_end(self):
        
        # log figure of current solution and ground truth
        if self.recursive:
            x_adp = self.x_iter
        else:
            x_adp = self.forward(self.x_iter).detach()
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr = 0.1)
    
class random_input(Dataset):
    # This class is not really used. The pytorch lightning trainer just gets it as an argument
    def __init__(self):
        self.X = torch.randn([1,10],dtype=torch.float)
    
    def __getitem__(self, idx):
        return self.X       
        
    def __len__(self):
        return 1
    
############################
#
# LISTA with infinite number of layers:
# after each training step, the output of the network becomes the input of the next step
#
###########################
unrolling = True
param_adp = np.array([[[0.087, 0.0145],[0.072,0.072],[0.072,0.024]],
                      [[0.108, 0.027],[0.042,0.168],[0.123,0.041]]])
adp_al1 = param_adp[forward][bsp][0]
adp_al2 = param_adp[forward][bsp][1]

xk=x_tik
xk = torch.zeros_like(xk)
R_el = y_norm2/4
lamb=1
if unrolling:
    model = ADP_eln(A, ydelta, xk, adp_al1, adp_al2, 1, True, x, x_eln, unroll_iter)
    data = random_input()

    trainer = Trainer(accelerator="cpu", max_epochs=4000)
    trainer.fit(model, DataLoader(data, batch_size=1))

    print(str('ready (') + str(model.k) + str(')'))

    # print model errors:
    error = {}
    for i in model.results_adp.keys():
        error[i] = round(((x.detach().numpy()-model.results_adp[i].reshape(x.shape).numpy())**2).sum()*h,4)

    print('ADP eln LISTA infinity')
    print(error)
    unroll_iter = model.unroll_iter
    
    plt.figure(figsize=(8, 8))
    plt.imshow(A.detach(), vmin=A.min(), vmax=A.max())
    plt.colorbar()  # Adding colorbar for A if needed
    plt.savefig('unroll_A.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Second plot: Heatmap of model.B
    plt.figure(figsize=(8, 8))
    plt.imshow(model.B.detach(), vmin=model.B.min(), vmax=model.B.max())
    plt.colorbar()
    plt.savefig('B_unroll.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # plot the results
    plt.figure(figsize=(8, 8))
    plt.plot(t, x, label='truth')
    plt.plot(t, model.final, label='ADP LISTA $L=\infty$')
    plt.plot(t, ydelta, label='Noisy data')
    plt.title(f"ADP-LISTA $L=\infty$ Error (l1): {(abs(model.final.reshape(x.shape) - x)).sum() * h:.3f}")
    plt.legend(loc='best')
    plt.savefig('Unroll_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

#################################
#
# ADP solved by implicit function theorem
#
#################################
Bk = A
#Bk = 1.1*torch.eye(xk.shape[0], dtype=A.dtype) - 0.1*A
xk = x_eln
xk = torch.zeros_like(xk)
al_adp_l1=adp_al1 
al_adp_l2=adp_al2 
lamb_B=0.1
lamb_x=0.1

res_xk = None
res_Bk = None
eye=torch.eye(xk.shape[0],dtype=xk.dtype)
loss = []
lower_iter = []
time = []
lower_iter.append(0)
lower_count_IFT = 0
loss.append(0.5* torch.linalg.norm(torch.matmul(A,xk)-ydelta)**2)
time.append(0)
time_start = T.time()
for k in range(1,7501):
    for j in range(500):
        #compute x(B) with a classical proximal gradient method
        xkn=xk-lamb_x*(torch.matmul(Bk.transpose(0,1),torch.matmul(Bk,xk)-ydelta))-lamb_x*al_adp_l2*xk
        xkn=proxl1(xkn,al_adp_l1*lamb_x)
        if (abs(xkn-xk)).sum()<0.000001:
            lower_count_IFT += (j+1)
            break
        xk=xkn
    lower_count_IFT += 500
    lower_iter.append(lower_count_IFT)
    #compute the gradient of x(B) w.r.t. B
    Axy = torch.matmul(A, xkn) - ydelta
    xi = torch.matmul(A.transpose(0,1),Axy)
    
    step = (1-al_adp_l2)*xkn - torch.matmul(Bk.transpose(0,1), torch.matmul(Bk,xkn)- ydelta) 
    
    IdBB = (1-al_adp_l2)*eye - torch.matmul(Bk.transpose(0,1),Bk)
    for i in range(step.shape[0]):
        if abs(step[i])<al_adp_l1:
            IdBB[i,:] = 0 
    
    v= torch.linalg.solve(eye - IdBB,xi)
    
    for i in range(step.shape[0]):
        if abs(step[i])<al_adp_l1:
            v[i,:] = 0 
    
    first = h*torch.matmul(torch.matmul(Bk,xk),v.transpose(0,1))
    second = h*torch.matmul(torch.matmul(Bk,v),xk.transpose(0,1))
    third = h*torch.matmul(ydelta,v.transpose(0,1))
    
    gradB = -first-second+third
    
    Bk = Bk - lamb_B*gradB
    with torch.no_grad():
        res_xk = xk
        res_Bk = Bk
    loss.append(0.5* torch.linalg.norm(torch.matmul(A,xk)-ydelta)**2)
    time.append(T.time()-time_start)

plt.figure(figsize=(8, 8))
plt.plot(t, x, label='truth')
plt.plot(t, res_xk, label='ADP IFT')
plt.plot(t, ydelta, label='Noisy data')
plt.title(f"ADP-IFT Error (l1): {(abs(res_xk.reshape(x.shape) - x)).sum() * h:.3f}")
plt.legend(loc='best')
plt.savefig('IFT_curve.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 8))
plt.imshow(res_Bk, vmin=res_Bk.min(), vmax=res_Bk.max())
plt.colorbar()
plt.savefig('B_IFT.png', dpi=300, bbox_inches='tight')
plt.close()

print('ADP-IF Error (l1): ' + str((abs(res_xk.reshape(x.shape)-x)).sum()*h))



# MAID
Bk = A
#Bk = 1.1*torch.eye(xk.shape[0], dtype=A.dtype) - 0.1*A
xk = x_eln
xk = torch.zeros_like(xk)
al_adp_l1=adp_al1 
al_adp_l2=adp_al2 
lamb_B=0.1
lamb_x=0.1

res_xk = None
res_Bk = None
eye=torch.eye(xk.shape[0],dtype=xk.dtype)
loss_MAID = []
lower_count = 0
lower_iter_MAID = []
time_MAID = []
lower_iter_MAID.append(0)
loss_MAID.append(0.5* torch.linalg.norm(torch.matmul(A,xk)-ydelta)**2)
time_MAID.append(0)
time_start = T.time()
eps = 1e-3
max_upper_iter = 300
upper_objective = lambda x: 0.5*torch.linalg.norm(torch.matmul(A,x)-ydelta)**2
def lower_solver(x, B, y, lamb_x, al_adp_l1, al_adp_l2, h, max_iter, eps, lower_count):
    for j in range(max_iter):
        #compute x(B) with a classical proximal gradient method
        xkn=x-lamb_x*(torch.matmul(B.transpose(0,1),torch.matmul(B,x)-y))-lamb_x*al_adp_l2*x
        xkn=proxl1(xkn,al_adp_l1*lamb_x)
        if torch.linalg.norm(torch.matmul(B,xkn)-y + al_adp_l1*torch.sign(xkn) + al_adp_l2*xkn) <= eps:
            lower_count += (j+1)
            print ("Lower solver: ", j)
            return xkn, lower_count
        x=xkn
    lower_count += max_iter
    return xkn, lower_count
def grad_B_func(x, B, y, al_adp_l2, h):
    Axy = torch.matmul(A, x) - y
    xi = torch.matmul(A.transpose(0,1),Axy)
    step = (1-al_adp_l2)*x - torch.matmul(B.transpose(0,1), torch.matmul(B,x)- y) 
    IdBB = (1-al_adp_l2)*eye - torch.matmul(B.transpose(0,1),B)
    for i in range(step.shape[0]):
        if abs(step[i])<al_adp_l1:
            IdBB[i,:] = 0 
    v= torch.linalg.solve(eye - IdBB,xi)
    for i in range(step.shape[0]):
        if abs(step[i])<al_adp_l1:
            v[i,:] = 0 
    first = h*torch.matmul(torch.matmul(B,x),v.transpose(0,1))
    second = h*torch.matmul(torch.matmul(B,v),x.transpose(0,1))
    third = h*torch.matmul(y,v.transpose(0,1))
    gradB = -first-second+third
    return gradB
rho = 0.5
tau = 0.5
nu = 1.05
step_increase = 10/9
beta = lamb_B 
eta = 1e-4
for k in range(max_upper_iter):
    xk, lower_count = lower_solver(xk, Bk, ydelta, lamb_x, al_adp_l1, al_adp_l2, h, 500, eps, lower_count)
    # line search loop
    for i in range(5):
        grad_B = grad_B_func(xk, Bk, ydelta, al_adp_l2, h)
        g_old = loss_MAID[-1]
        B_new = Bk-beta * (rho**i)*grad_B
        x_new,lower_count = lower_solver(xk, B_new, ydelta, lamb_x, al_adp_l1, al_adp_l2, h, 500, eps, lower_count)
        g = upper_objective(x_new)
        grad_upper = lambda x: torch.matmul(A.T, torch.matmul(A,x)-ydelta)
        if g + torch.linalg.norm(grad_upper(x_new))* eps +  eps**2/2 - g_old + torch.linalg.norm(grad_upper(xk))* eps \
                    <= -beta * rho**i * eta*(eta) * torch.linalg.norm(grad_B)**2:
            Bk = B_new
            xk = x_new
            eps *= nu
            beta = beta * rho**i
            beta *= step_increase
            print("Backtracking line search: ", i)
            break
    eps *= tau
    with torch.no_grad():
        res_xk = xk
        res_Bk = Bk
    loss_MAID.append(0.5* torch.linalg.norm(torch.matmul(A,xk)-ydelta)**2)
    lower_iter_MAID.append(lower_count)
    time_MAID.append(T.time()-time_start)
    if ((loss_MAID[-2] - loss_MAID[-1])) < 1e-12 and (loss_MAID[-2] - loss_MAID[-1]) != 0 and k>10:
        break


plt.figure(figsize=(8, 8))
plt.plot(t, x, label='truth')
plt.plot(t, res_xk, label='ADP MAID')
plt.plot(t, ydelta, label='Noisy data')
plt.title(f"ADP-MAID Error (l1): {(abs(res_xk.reshape(x.shape) - x)).sum() * h:.3f}")
plt.legend(loc='best')
plt.savefig('MAID_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# Second plot: Heatmap
plt.figure(figsize=(8, 8))
plt.imshow(res_Bk, vmin=res_Bk.min(), vmax=res_Bk.max())
plt.colorbar()
plt.savefig('B_MAID.png', dpi=300, bbox_inches='tight')
plt.close()
logs = torch.load('1D_logs.pth')
time = logs["time_IFT"]
loss = logs["loss_IFT"]
time_MAID = logs["time_MAID"]
loss_MAID = logs["loss_MAID"]
time_unroll = logs["time_unroll"]
loss_unroll = logs["loss_unroll"]
unroll_iter = logs["iter_unroll"]
lower_iter = logs["iter_IFT"]
lower_iter_MAID = logs["iter_MAID"]

# Time and computational cost comparison
plt.plot(time_MAID,loss_MAID,label='MAID')
plt.plot(time,loss,label='High accuracy IFT')
plt.plot(time_unroll,loss_unroll,label='LISTA $L=\infty$')
plt.legend(fontsize=20)
plt.yscale('log')   
plt.xlabel('time [s]')
plt.ylabel('Upper-level loss')
plt.savefig('1D_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot the number of iterations
lower_iter_MAID[0] = 1
lower_iter[0] = 1
unroll_iter[0] = 1
plt.plot(lower_iter_MAID, loss_MAID,label='MAID')
plt.plot(lower_iter, loss,label='High accuracy IFT')
plt.plot(unroll_iter, loss_unroll,label='LISTA $L=\infty$')
plt.legend(fontsize=20)
plt.yscale('log')
plt.xscale('log')
plt.xticks([1e1, 1e3, 1e5, 1e6])
plt.xlabel('Lower-level iterations')
plt.ylabel('Upper-level loss')
plt.savefig('1D_comparison_iter.png', dpi=300, bbox_inches='tight')
plt.close()

# Save logs
# logs ={"iter_MAID": lower_iter_MAID, "loss_MAID": loss_MAID, "time_MAID": time_MAID, "iter_unroll": unroll_iter, "loss_unroll": loss_unroll, "time_unroll": time_unroll, "iter_IFT": lower_iter, "loss_IFT": loss, "time_IFT": time}
# torch.save(logs, '1D_logs.pth')