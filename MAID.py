import torch
from torch.optim import SGD

class MAID(SGD):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, eps = 1e-1, 
                 nu_acc = 1.05, nu_step = 1.5, tau = 0.5, rho = 0.1, max_line_search = 3, 
                 hypergrad_module = None, fixed_eps = False, fixed_lr = False):
        """
        Custom SGD optimizer that only updates parameters if a certain condition is satisfied.

        :param params: Model parameters to optimize.
        :param lr: Learning rate.
        :param momentum: Momentum factor.
        :param dampening: Dampening for momentum.
        :param weight_decay: L2 penalty (weight decay).
        :param nesterov: Enables Nesterov momentum.
        :param condition_func: A function that checks whether to accept the update.
        :param eps: accuracy of lower level optimization
        """
        super(MAID, self).__init__(params, lr, momentum, dampening,weight_decay, nesterov)
        self.state['successful'] = False
        self.eps = eps # Accuracy of lower level optimization
        self.eps_old = eps
        self.nu_acc = nu_acc # increase factor for accuracy
        self.nu_step = nu_step # increase factor for step size
        self.tau = tau  # decrease factor for accuracy
        self.rho = rho  # decrease factor for step size
        self.lr = lr # initial step size
        self.max_line_search = max_line_search
        self.lower_solver = hypergrad_module.FISTA
        self.loss = hypergrad_module.upper_level_obj
        self.grad_loss = hypergrad_module.grad_upper_level
        self.hypergrad = hypergrad_module
        self.loss_old = torch.inf
        self.old_step = None
        self.fixed_eps = fixed_eps
        self.fixed_lr = fixed_lr
    def linesearch(self, grad_params, old_params):
        """
        line search mechanism of MAID
        """
        Lg = 1/(self.hypergrad.x_init.shape[0]* self.hypergrad.x_init.shape[1])
        flattened_grads = torch.cat([g.reshape(-1) for g in grad_params])
        for i in range(self.max_line_search):
            # loss_old = self.loss(self.hypergrad.x_init)
            loss_old = self.loss_old
            self.lr = self.lr * self.rho**i
            super(MAID, self).step()
            # self.hypergrad.lower_level_obj.regularizer.zero_mean()
            x_new = self.hypergrad.FISTA(self.hypergrad.x_init, self.eps, self.hypergrad.max_lower_iter).detach() 
            if (self.loss(x_new) + torch.norm(self.grad_loss(x_new)) * self.eps/(self.hypergrad.x_init.shape[0] * self.hypergrad.x_init.shape[1]) + Lg *  self.eps**2/2 - \
                loss_old + torch.norm(self.grad_loss(self.hypergrad.x_init)) * self.eps_old /(self.hypergrad.x_init.shape[0]* self.hypergrad.x_init.shape[1]) \
                    <= -self.lr * 1e-8 * torch.norm(flattened_grads)**2) or self.loss(x_new) - loss_old <= -self.lr * 1e-8 * torch.norm(flattened_grads)**2:
                print("loss", self.loss(x_new).item(), f'eps: {self.eps}', f'lr: {self.lr}', f'iter: {i}')
                self.hypergrad.x_init = x_new
                self.loss_old = self.loss(x_new)
                return True , x_new.detach(), self.loss(x_new)       
            self.revert(old_params)
            self.lr = self.old_step
            print("loss old", loss_old.item(), "loss", self.loss(x_new).item(), f'eps: {self.eps}', f'lr: {self.lr}', f'iter: {i}')
        return False , self.hypergrad.x_init, loss_old
    def revert(self, params_before):
        # Revert to the old parameters if condition not met
        for i, param in enumerate(self.param_groups[0]['params']):
            if param.grad is not None:
                param.data = params_before[i].data
    def step(self, closure=None):
        """
        Performs a conditional parameter update.
        The parameters are updated only if `condition_func` returns True.
        """
        # self.x_old = self.hypergrad.x_init
        loss = None
        if closure is not None:
            loss = closure()
        self.old_step = self.lr
        self.eps_old = self.eps            
        # Save current parameters
        params_before = [p.clone() for p in self.param_groups[0]['params'] if p.grad is not None]

        # gradient of parameters
        grad_params = [p.grad for p in self.param_groups[0]['params'] if p.grad is not None]
        
        # # Perform the standard SGD step (provisionally)
        # super(MAID, self).step()

        # Get updated parameters
        params_after = [p.clone() for p in self.param_groups[0]['params'] if p.grad is not None]
        if not self.fixed_lr:
            success, x_new , loss = self.linesearch(grad_params, params_before)
        else:
            success = True
            super(MAID, self).step()
            # x_new = self.hypergrad.FISTA(self.hypergrad.x_init, self.eps, self.hypergrad.max_lower_iter).detach()
            x_new = self.hypergrad.x_init.detach()
            loss = self.loss(x_new)

        # Check the condition function
        if not success:
            # self.revert(params_before)
            # self.hypergrad.x_init = self.x_old
            self.state['successful'] = False
            if not self.fixed_eps:
                self.eps = self.eps * self.tau
            self.hypergrad.lower_tol = self.eps
            self.hypergrad.cg_tol = self.eps
        else:
            # Update was successful
            self.state['successful'] = True
            if not self.fixed_eps:
                self.eps = self.eps * self.nu_acc
            if not self.fixed_lr:
                self.lr = self.lr * self.nu_step
            self.hypergrad.lower_tol = self.eps
            self.hypergrad.cg_tol = self.eps
        return loss , x_new


