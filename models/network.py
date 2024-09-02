import logging
from typing import List
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from pde.pde import omega, scale_back_t

class PINN(nn.Module):
    def __init__(self, args, train_inputs, train_labels,
                             valid_inputs, valid_labels):
        super().__init__()

        self.args = args
        self.layers = np.array(args.layers)
        self.n_output = self.layers[-1]
        self.pinn = args.pinn
        self.device = args.device
        self.IC_weight = args.IC_weight
        self.BC_weight = args.BC_weight
        self.PDE_weight = args.PDE_weight
        self.activation = nn.Tanh()

        # Inputs
        if args.hamiltonian == "input":
            self.train_ic_pts, self.train_col_pts = train_inputs
            self.valid_ic_pts, self.valid_col_pts = valid_inputs
            self.train_ic_phi = train_labels[0]
            self.valid_ic_phi = valid_labels[0]
        else:
            self.train_ic_pts, self.train_bc_pts, self.train_col_pts = train_inputs
            self.valid_ic_pts, self.valid_bc_pts, self.valid_col_pts = valid_inputs
            self.train_ic_phi, self.train_bc_ham = train_labels
            self.valid_ic_phi, self.valid_bc_ham = valid_labels

        # loss function
        self.loss_function = nn.MSELoss(reduction="mean")

        # fully connected layers
        self.net = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        self.init_weights()

        if args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate)
        elif args.optimizer == "l-bfgs":
            self.optimizer = torch.optim.LBFGS(self.net.parameters(), lr=args.learning_rate, max_iter=args.epochs, max_eval=5000,
                                               history_size=args.history_size, tolerance_grad=args.tolerance_grad, tolerance_change=args.tolerance_change,
                                               line_search_fn="strong_wolfe")
        else:
            raise ValueError("Unknown optimizer")

        self.iter = 0
        self.train_loss = 0.0
        self.valid_loss = 0.0
        self.train_ic_loss = 0.0
        self.valid_ic_loss = 0.0
        self.train_bc_loss = 0.0
        self.valid_bc_loss = 0.0
        self.train_pde_loss = 0.0
        self.valid_pde_loss = 0.0
        
        self.iter_list = []
        self.train_loss_list = []
        self.valid_loss_list = []
        self.train_ic_ll = []
        self.valid_ic_ll = []
        self.train_bc_ll = []
        self.valid_bc_ll = []
        self.train_pde_ll = []
        self.valid_pde_ll = []

        # feature scaling
        self.q_min, self.q_max = args.q_min_max
        self.p_min, self.p_max = args.p_min_max
        self.t_min, self.t_max = args.t_min_max
        if args.hamiltonian == "input":
            self.h_min, self.h_max = args.h_min_max
            self.lb = torch.Tensor([self.q_min, self.p_min, self.t_min, self.h_min]).reshape(1, -1).to(self.device)
            self.ub = torch.Tensor([self.q_max, self.p_max, self.t_max, self.h_max]).reshape(1, -1).to(self.device)
        else:
            self.lb = torch.Tensor([self.q_min, self.p_min, self.t_min]).reshape(1, -1).to(self.device)
            self.ub = torch.Tensor([self.q_max, self.p_max, self.t_max]).reshape(1, -1).to(self.device)

    def init_weights(self):
        # Xavier Normal Initialization
        for i in range(len(self.layers)-1):
            nn.init.xavier_normal_(self.net[i].weight.data, gain=1.0)

            # set biases to zero
            nn.init.zeros_(self.net[i].bias.data)
    
    def forward(self, q, p, t, h):
        if self.args.hamiltonian == "input":
            u = torch.cat((q, p, t, h), dim = 1)
        else:
            u = torch.cat((q, p, t), dim = 1)

        u = (u - self.lb) / (self.ub - self.lb)
        a = u.float()

        for i in range(len(self.layers)-2):
            z = self.net[i](a)
            a = self.activation(z)

        return self.net[-1](a)

    def loss_IC(self, q, p, t, h, IC):
        return self.loss_function(self.forward(q, p, t, h)[:, 0].reshape(-1, 1), IC)

    def loss_BC(self, q, p, t, h, BC):
        return self.loss_function(self.forward(q, p, t, h)[:, 1].reshape(-1, 1), BC)

    def loss_PDE(self, q, p, t, h):
        q.requires_grad = True
        p.requires_grad = True
        t.requires_grad = True
        if self.args.hamiltonian == "input":
            h.requires_grad = True

        f = self.forward(q, p, t, h)
    
        # Construct the PDE loss using torch.autograd (THIS IS THE KEY)
        f_q = autograd.grad(f, q,
                            create_graph = True,
                            grad_outputs = torch.ones_like(f).to(self.device),
                            allow_unused = True)[0]
        f_t = autograd.grad(f, t,
                            create_graph = True,
                            grad_outputs = torch.ones_like(f).to(self.device),
                            allow_unused = True)[0]

        if self.args.hamiltonian == "output":
            h_q = autograd.grad(h, q,
                                create_graph = True,
                                grad_outputs = torch.ones_like(h).to(self.device),
                                allow_unused = True)[0]
            h_p = autograd.grad(h, p,
                                create_graph = True,
                                grad_outputs = torch.ones_like(h).to(self.device),
                                allow_unused = True)[0]

            LHS_1 = h_q - q * 0.0 / self.args.w0 ** 2
            LHS_2 = h_p - p
            loss1 = self.loss_function(LHS_1, torch.zeros(h.shape).to(self.device))
            loss2 = self.loss_function(LHS_2, torch.zeros(h.shape).to(self.device))

        LHS = 2 * np.pi * self.args.N0 * f_t + p * f_q
        loss = self.loss_function(LHS, torch.zeros(f.shape).to(self.device))

        return loss

    def compute_loss(self, valid=False):
        ic_q, ic_p, ic_t, ic_h = self.get_inputs(self.train_ic_pts if not valid else self.valid_ic_pts)
        ic_label = self.train_ic_phi.reshape(-1, 1) if not valid else self.valid_ic_phi.reshape(-1, 1)

        if self.args.hamiltonian == "output":
            bc_q, bc_p, bc_t, bc_h = self.get_inputs(self.train_bc_pts if not valid else self.valid_bc_pts)
            bc_label = self.train_bc_ham.reshape(-1, 1) if not valid else self.valid_bc_ham.reshape(-1, 1)

        if self.pinn:
            col_q, col_p, col_t, col_h = self.get_inputs(self.train_col_pts if not valid else self.valid_col_pts)

        # Compute losses

        IC_loss = self.loss_IC(ic_q, ic_p, ic_t, ic_h, ic_label)
        BC_loss = self.loss_BC(bc_q, bc_p, bc_t, bc_h, bc_label) if self.args.hamiltonian == "output" else 0.0
        PDE_loss = self.loss_PDE(col_q, col_p, col_t, col_h) if self.pinn else 0.0
        total_loss = self.IC_weight * IC_loss + self.BC_weight * BC_loss + self.PDE_weight * PDE_loss

        return total_loss, IC_loss, BC_loss, PDE_loss
    
    def closure(self):

        self.optimizer.zero_grad()
        self.train_loss, self.train_ic_loss, self.train_bc_loss, self.train_pde_loss = self.compute_loss()

        if self.iter % self.args.log_freq == 0:
            self.net.eval()
            self.valid_loss, self.valid_ic_loss, self.valid_bc_loss, self.valid_pde_loss = self.compute_loss(valid=True)
            self.log_losses()

            self.net.train()
        
        self.train_loss.backward()
        self.iter += 1
        
        return self.train_loss
    
    def train_with_lbfgs(self):
        self.net.train()
        self.optimizer.step(self.closure)

    def train_with_adam(self):
        for iter in range(1, self.args.epochs + 1):

            self.iter = iter

            # Training phase
            self.net.train()
            self.optimizer.zero_grad()
            self.train_loss, self.train_ic_loss, self.train_bc_loss, self.train_pde_loss = self.compute_loss()

            if self.args.log_loss:
                if self.iter % self.args.log_freq == 0:
                    self.net.eval()
                    self.valid_loss, self.valid_ic_loss, self.valid_bc_loss, self.valid_pde_loss = self.compute_loss(valid=True)
                    self.log_losses()

            self.train_loss.backward()
            self.optimizer.step()

    def log_losses(self):
        print(f"Epoch {self.iter}/{self.args.epochs}, train loss: {self.train_loss.item():.9f}, valid loss: {self.valid_loss.item():.9f}")
        logging.info(f"Epoch {self.iter}/{self.args.epochs}, train loss: {self.train_loss.item():.9f}, valid loss: {self.valid_loss.item():.9f}")
        self.iter_list.append(self.iter)
        self.train_loss_list.append(self.train_loss.item())
        self.valid_loss_list.append(self.valid_loss.item())
        self.train_ic_ll.append(self.train_ic_loss.item())
        self.valid_ic_ll.append(self.valid_ic_loss.item())
        self.train_bc_ll.append(self.train_bc_loss.item() if self.args.hamiltonian == "output" else 0.0)
        self.valid_bc_ll.append(self.valid_bc_loss.item() if self.args.hamiltonian == "output" else 0.0)
        self.train_pde_ll.append(self.train_pde_loss.item())
        self.valid_pde_ll.append(self.valid_pde_loss.item())

    def get_inputs(self, data_pts):
        q = data_pts[:, 0].reshape(-1, 1)
        p = data_pts[:, 1].reshape(-1, 1)
        t = data_pts[:, 2].reshape(-1, 1)
        if self.args.hamiltonian == "input":
            h = data_pts[:, 3].reshape(-1, 1)
        else:
            h = torch.zeros(q.shape).to(self.device)
        
        return q, p, t, h