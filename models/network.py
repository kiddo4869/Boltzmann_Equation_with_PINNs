from typing import List
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

class PINN(nn.Module):
    def __init__(self, args, X_train, y_train, X_train_with_col,
                             X_valid, y_valid, X_valid_with_col):
        super().__init__()

        # Inputs
        self.X_train = X_train
        self.y_train = y_train
        self.X_train_with_col = X_train_with_col
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_valid_with_col = X_valid_with_col

        self.args = args
        self.layers = np.array(args.layers)
        self.n_output = self.layers[-1]
        self.pinn = args.pinn
        self.device = args.device
        self.PDE_weight = args.PDE_weight
        self.activation = nn.Tanh()

        # loss function
        self.loss_function = nn.MSELoss(reduction="mean")

        # fully connected layers
        self.net = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        self.init_weights()

        if args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate)
        elif args.optimizer == "l-bfgs":
            self.optimizer = torch.optim.LBFGS(self.net.parameters(), lr=args.learning_rate, max_iter=args.epochs, max_eval=5000,
                                               history_size=100, tolerance_grad=1e-05, tolerance_change=1e-09,
                                               line_search_fn="strong_wolfe")
        else:
            raise ValueError("Unknown optimizer")

        self.iter = 0
        self.train_loss = 0.0
        self.valid_loss = 0.0
        self.train_ic_loss = 0.0
        self.valid_ic_loss = 0.0
        self.train_pde_loss = 0.0
        self.valid_pde_loss = 0.0
        
        self.iter_list = []
        self.train_loss_list = []
        self.valid_loss_list = []
        self.train_ic_ll = []
        self.valid_ic_ll = []
        self.train_pde_ll = []
        self.valid_pde_ll = []

        # feature scaling
        self.q_min, self.q_max = args.q_min_max
        self.p_min, self.p_max = args.p_min_max
        self.t_min, self.t_max = args.t_min_max
        self.lb = torch.Tensor([self.q_min, self.p_min, self.t_min]).to(self.device)
        self.ub = torch.Tensor([self.q_max, self.p_max, self.t_max]).to(self.device)
    
    def init_weights(self):
        # Xavier Normal Initialization
        for i in range(len(self.layers)-1):
            nn.init.xavier_normal_(self.net[i].weight.data, gain=1.0)

            # set biases to zero
            nn.init.zeros_(self.net[i].bias.data)

    def forward(self, q, p, t):
        u = torch.cat((q, p, t), dim = 1)
        u = (u - self.lb) / (self.ub - self.lb)
        a = u.float()

        for i in range(len(self.layers)-2):
            z = self.net[i](a)
            a = self.activation(z)

        return self.net[-1](a)

    def loss_IC(self, q, p, t, exact_sol):
        return self.loss_function(self.forward(q, p, t), exact_sol)
    
    def loss_PDE(self, q_col, p_col, t_col):
        q = q_col.clone().detach()
        p = p_col.clone().detach()
        t = t_col.clone().detach()

        q.requires_grad = True
        p.requires_grad = True
        t.requires_grad = True

        if not self.args.hamiltonian:
            f = self.forward(q, p, t)
        else:
            pred = self.forward(q, p, t)
            f, h = pred[:, 0].reshape(-1, 1), pred[:, 1].reshape(-1, 1)
        
        # Construct the PDE loss using torch.autograd (THIS IS THE KEY)
        f_q = autograd.grad(f, q,
                            create_graph = True,
                            grad_outputs = torch.ones_like(f).to(self.device),
                            allow_unused = True)[0]
        f_t = autograd.grad(f, t,
                            create_graph = True,
                            grad_outputs = torch.ones_like(f).to(self.device),
                            allow_unused = True)[0]

        LHS = 2 * np.pi * self.args.N0 * f_t + p * f_q

        if self.args.hamiltonian:
            h_q = autograd.grad(h, q,
                                create_graph = True,
                                grad_outputs = torch.ones_like(h).to(self.device),
                                allow_unused = True)[0]
            h_p = autograd.grad(h, p,
                                create_graph = True,
                                grad_outputs = torch.ones_like(h).to(self.device),
                                allow_unused = True)[0]

            LHS_1 = h_q * (h + 1) - q * self.args.k * self.args.T * 0.0 / self.args.w0
            LHS_2 = h_p * (h + 1) - p * self.args.k * self.args.T
            LHS += LHS_1 + LHS_2

        return self.loss_function(LHS, torch.zeros(f.shape).to(self.device))

    def compute_loss(self, valid=False):

        if not valid:
            q = self.X_train[:, 0].reshape(-1, 1)
            p = self.X_train[:, 1].reshape(-1, 1)
            t = self.X_train[:, 2].reshape(-1, 1)
            exact_sol = self.y_train.reshape(-1, self.n_output)
        else:
            q = self.X_valid[:, 0].reshape(-1, 1)
            p = self.X_valid[:, 1].reshape(-1, 1)
            t = self.X_valid[:, 2].reshape(-1, 1)
            exact_sol = self.y_valid.reshape(-1, self.n_output)

        if not self.pinn:
            IC_loss = self.loss_IC(q, p, t, exact_sol)
            return IC_loss, IC_loss, 0.0
        else:
            if not valid:
                q_col = self.X_train_with_col[:, 0].reshape(-1, 1)
                p_col = self.X_train_with_col[:, 1].reshape(-1, 1)
                t_col = self.X_train_with_col[:, 2].reshape(-1, 1)
            else:
                q_col = self.X_valid_with_col[:, 0].reshape(-1, 1)
                p_col = self.X_valid_with_col[:, 1].reshape(-1, 1)
                t_col = self.X_valid_with_col[:, 2].reshape(-1, 1)

            # Compute losses
            PDE_loss = self.loss_PDE(q_col, p_col, t_col)
            IC_loss = self.loss_IC(q, p, t, exact_sol)
            total_loss = (1 - self.PDE_weight) * IC_loss + self.PDE_weight * PDE_loss

            return total_loss, IC_loss, PDE_loss
    
    def closure(self):

        self.optimizer.zero_grad()
        self.train_loss, self.train_ic_loss, self.train_pde_loss = self.compute_loss()

        if self.iter % self.args.log_freq == 0:
            self.net.eval()
            self.valid_loss, self.valid_ic_loss, self.valid_pde_loss = self.compute_loss(valid=True)
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
            self.train_loss, self.train_ic_loss, self.train_pde_loss = self.compute_loss()

            if self.args.log_loss:
                if self.iter % self.args.log_freq == 0:
                    self.net.eval()
                    self.valid_loss, self.valid_ic_loss, self.valid_pde_loss = self.compute_loss(valid=True)
                    self.log_losses()

            self.train_loss.backward()
            self.optimizer.step()

    def log_losses(self):
        print(f"Epoch {self.iter}/{self.args.epochs}, train loss: {self.train_loss.item():.9f}, valid loss: {self.valid_loss.item():.9f}")
        self.iter_list.append(self.iter)
        self.train_loss_list.append(self.train_loss.item())
        self.valid_loss_list.append(self.valid_loss.item())
        self.train_ic_ll.append(self.train_ic_loss.item())
        self.valid_ic_ll.append(self.valid_ic_loss.item())
        self.train_pde_ll.append(self.train_pde_loss.item())
        self.valid_pde_ll.append(self.valid_pde_loss.item())