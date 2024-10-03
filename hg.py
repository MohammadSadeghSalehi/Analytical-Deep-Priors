import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

class Hypergrad_Calculator(nn.Module):
    def __init__(self, lower_level_obj, upper_level_obj, x_init = None, \
        lower_tol = 1e-1, cg_tol = 1e-1, max_lower_iter = 200, max_cg_iter = 200,\
        warm_start = True, verbose = False, lipschitz_estimate = True, \
        grad_fidelity = None, grad_reg = None, grad_upper = None, hessian = None, jac = None, grad_normalize = False, lower_skip = False):
        super(Hypergrad_Calculator, self).__init__()
        self.lower_level_obj = lower_level_obj
        self.upper_level_obj = upper_level_obj
        self.x_init = x_init
        self.lower_tol = lower_tol
        self.cg_tol = cg_tol
        self.max_lower_iter = max_lower_iter
        self.max_cg_iter = max_cg_iter
        self.warm_start = warm_start
        self.verbose = verbose
        self.lipschitz_estimate = lipschitz_estimate
        self.grad_fidelity = grad_fidelity
        self.grad_reg = grad_reg
        self.grad_upper = grad_upper
        self.hessian = hessian
        self.jac = jac
        self.grad_normalize = grad_normalize
        self.hvpi = None
        self.logs = {"lower_counter": 0, "cg_counter": 0}
        self.lower_skip = lower_skip
    def grad_lower_level(self, x):
        if self.grad_fidelity is not None and self.grad_reg is not None:
            return self.grad_fidelity(x) + self.grad_reg(x)
        elif self.grad_fidelity is not None:
            return self.grad_fidelity(x) + torch.autograd.grad(self.lower_level_obj.regularizer(x), x, create_graph=True)[0]
        elif self.grad_reg is not None:
            return torch.autograd.grad(self.lower_level_obj.data_fidelity(x), x, create_graph=True)[0] + self.grad_reg(x)
        grad_fidelity = torch.autograd.grad(self.lower_level_obj.data_fidelity(x), x, create_graph=True , retain_graph= True)[0]
        grad_reg = torch.autograd.grad(self.lower_level_obj.regularizer(x), x, create_graph=True, retain_graph= True)[0]
        return grad_fidelity + grad_reg
    def grad_upper_level(self, x):
        x = x.requires_grad_(True)
        if self.grad_upper is not None:
            return self.grad_upper(x)
        return torch.autograd.grad(self.upper_level_obj(x), x, create_graph=False)[0].detach()
    # def cg(self, A, b, x0, tol, max_iter):
    #     x = x0
    #     r = b - A(x)
    #     p = r
    #     rsold = torch.norm(r)**2
    #     for i in range(max_iter):
    #         Ap = A(p)
    #         alpha = rsold / torch.dot(p.view(-1), Ap.view(-1))
    #         x = x + alpha * p
    #         r = r - alpha * Ap
    #         rsnew = (torch.norm(r , dim = (2, 3))**2).mean()
    #         if torch.sqrt(rsnew) < tol:
    #             break
    #         p = r + (rsnew / rsold) * p
    #         rsold = rsnew
    #     if self.verbose:
    #         print("CG iterations: ", i, "residual: ", torch.sqrt(rsnew).item())
    #     return x

    def cg(self, A, b, x0, tol, max_iter):
        x = x0
        r = b - A(x)
        p = r
        rsold = torch.norm(r.view(r.size(0), -1), dim=1)**2
        
        for i in range(max_iter):
            Ap = A(p)
            alpha = rsold / torch.sum(p.view(p.size(0), -1) * Ap.view(Ap.size(0), -1), dim=1)
            alpha = alpha.view(-1, 1, 1, 1)  # Make alpha broadcastable
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.norm(r.view(r.size(0), -1), dim=1)**2
            
            if torch.sqrt(torch.max(rsnew)) < tol * x.shape[0]:
                break
 
            p = r + (rsnew / rsold).view(-1, 1, 1, 1) * p
            rsold = rsnew
        self.logs["cg_counter"] += (i+1) * x.shape[0]
        print("CG iterations: ", i + 1, "max residual: ", torch.sqrt(torch.max(rsnew)).item())
        return x.detach()
    def power_method(self, A, x0, max_iter):
        x = x0
        for i in range(max_iter):
            x = A(x)
            x = x / torch.norm(x)
        return torch.dot(A(x).view(-1), x.view(-1)) / torch.dot(x.view(-1), x.view(-1))
    def LBFGS(self, x, tol, max_iter):
        x.requires_grad = True
        if self.lipschitz_estimate:
            L = self.power_method(lambda input: self.hessian_vector_product(x, input), torch.randn_like(x), 10)
            L = L.item()
            optimizer = optim.LBFGS([x], lr=1/L, max_iter=max_iter, tolerance_grad=tol, line_search_fn='strong_wolfe')
        else:
            optimizer = optim.LBFGS([x], lr=1, max_iter=max_iter, tolerance_grad=tol, line_search_fn='strong_wolfe') 
        def closure():
            optimizer.zero_grad()
            loss = self.lower_level_obj(x)
            loss.backward()
            if self.verbose:
                print("lower-level loss: ", loss.item(), "gradient norm: ", torch.norm(x.grad).item())
            if torch.norm(x.grad)/x.shape[0] < tol:
                return loss
            return loss
        optimizer.step(closure)
        return x.detach()
    def FISTA(self, x, tol, max_iter):
        # Initialize variables
        x = x.clone().detach()
        y = x.clone().detach()
        t = 1

        if self.lipschitz_estimate:
            L = self.power_method(lambda input: self.hessian_vector_product(x, input), torch.randn_like(x), 10)
            L = L.item()
        else:
            L = 1.0
        for i in range(max_iter):
            y.requires_grad_(True)  # Ensure y has gradients enabled for each iteration

            # Compute gradient
            grad = self.grad_lower_level(y)
            with torch.no_grad():
                x_new = y - (1/L) * grad  # Gradient step

                # Nesterov acceleration step
                t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
                y = x_new + ((t - 1) / t_new) * (x_new - x)

                # Update x and t
                x.copy_(x_new)
                t = t_new

            # Check convergence
            norm_inner = torch.norm(grad, dim = (2, 3))
            norm_channel = torch.mean(norm_inner, dim = 1)
            if (norm_channel.max() < tol / x.shape[0]):
                if self.verbose:
                    print(f"Converged at iteration {i + 1}, gradient norm: {torch.norm(grad).item()}")
                break

            if self.verbose:
                loss = self.lower_level_obj(x).item()
                grad_norm = torch.norm(grad).item()
                print(f"Iteration {i + 1}, loss: {loss}, gradient norm: {grad_norm}")
        self.logs["lower_counter"] += (i+1) * x.shape[0]
        return x.detach()
    def FISTA_elementwise(self, x, tol, max_iter):
        # Initialize variables
        x = x.clone().detach()
        y = x.clone().detach()
        t = torch.ones(x.shape[0], device=x.device)  # One 't' value per batch element

        # Estimate or use constant Lipschitz value
        if self.lipschitz_estimate:
            # Use the same L for all elements, or estimate L for each batch element
            L = torch.tensor([self.power_method(lambda input: self.hessian_vector_product(x[j], input), torch.randn_like(x[j]), 10) 
                            for j in range(x.shape[0])], device=x.device)
        else:
            L = torch.ones(x.shape[0], device=x.device)

        for i in range(max_iter):
            y.requires_grad_(True)  # Enable gradient computation for y

            # Compute the gradient for all elements in batch at once
            grad = self.grad_lower_level(y)  # Assuming this function works on batches

            with torch.no_grad():
                # Update all batch elements in parallel
                x_new = y - grad / L.view(-1, 1, 1, 1)  # Vectorized step with broadcast for L
                t_new = (1 + torch.sqrt(1 + 4 * t**2)) / 2

                # Update y and x in a vectorized manner
                y = x_new + ((t - 1) / t_new).view(-1, 1, 1, 1) * (x_new - x)
                x.copy_(x_new)
                t = t_new

            # Compute gradient norm for stopping criterion
            norm_inner = torch.norm(grad.view(grad.shape[0], -1), dim=-1)  # Vectorized gradient norm over batch

            # Check stopping condition for each batch element
            if torch.all(norm_inner < tol):
                if self.verbose:
                    print(f"Converged at iteration {i + 1}, gradient norm: {norm_inner}")
                self.logs["lower_counter"] += (i + 1) * x.shape[0]  # Log the number of iterations per batch element
                break

        # If no early stop, log the total number of iterations
        self.logs["lower_counter"] += (i + 1) * x.shape[0]  
        return x


            
        
    def hessian_vector_product(self, x, v):
        return torch.autograd.functional.vhp(self.lower_level_obj, x, v)[1].detach()
        # x = x.requires_grad_(True)
        # dot = torch.dot(self.grad_lower_level(x).view(-1), v.view(-1))
        # return torch.autograd.grad(dot, x, create_graph=True)[0].detach()
    def jac_vector_product(self, x, v):
        # computing sobolov grad
        self.upper_level_obj.kernel = self.lower_level_obj.regularizer.forward_operator.conv.weight.data
        _, sobolov_grad = self.upper_level_obj.sobolev_norm()
        for param in self.lower_level_obj.regularizer.parameters():
            if param.requires_grad:
                dot = torch.dot(self.grad_lower_level(x).view(-1), v.view(-1))
                if self.grad_normalize:
                    grad_param = torch.autograd.grad(dot, param, create_graph=True)[0]
                    param.grad =  - grad_param/ torch.norm(grad_param)
                else:
                    param.grad =  - torch.autograd.grad(dot, param, create_graph=False)[0].detach()
        for param in self.lower_level_obj.regularizer.forward_operator.parameters():
            param.grad += sobolov_grad
        return 
    def hypergrad(self):
        if not self.lower_skip:
            x_new = self.FISTA(self.x_init, self.lower_tol, self.max_lower_iter).requires_grad_(True)
        else:
            x_new = self.x_init
        # x_new = self.FISTA_elementwise(self.x_init, self.lower_tol, self.max_lower_iter).requires_grad_(True)
        if self.warm_start:
            self.x_init = x_new.detach()
        rhs = self.grad_upper_level(x_new)
        if self.hessian is None:
            if self.hvpi is None:
                q = self.cg(lambda input: self.hessian_vector_product(x_new, input), rhs, torch.randn_like(rhs), self.cg_tol, self.max_cg_iter)
                self.hvpi = q
            else:
                q = self.cg(lambda input: self.hessian_vector_product(x_new, input), rhs, self.hvpi, self.cg_tol, self.max_cg_iter)
        else:
            q = self.cg(lambda input: self.hessian(x_new, input), rhs, torch.randn_like(rhs), self.cg_tol, self.max_cg_iter)
        if self.jac is None:
            self.jac_vector_product(x_new, q)
        else:
            self.jac(x_new, q)