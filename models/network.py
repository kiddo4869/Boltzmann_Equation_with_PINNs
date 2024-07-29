from typing import List
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

class PINN(nn.Module):
    def __init__(self, args, X_train, y_train, X_train_with_col):
        super().__init__()

        # Inputs
        self.X_train = X_train
        self.y_train = y_train
        self.X_train_with_col = X_train_with_col

        self.args = args
        self.layers = np.array(args.layers)
        self.pinn = args.pinn
        self.device = args.device
        self.activation = nn.Tanh()
        self.weight = 0.25

        # loss function
        self.loss_function = nn.MSELoss(reduction="mean")

        # fully connected layers
        self.net = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        self.init_weights()
        
        self.optimizer = torch.optim.LBFGS(self.net.parameters(), lr=0.01, max_iter=2000, max_eval=5000,
                                           history_size=100, tolerance_grad=1e-05, tolerance_change=1e-09,
                                           line_search_fn="strong_wolfe")

        self.iter = 0
        self.loss = 0.0
        
        self.iter_list = []
        self.loss_list = []

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

    def loss_IC(self, q, p, t, phi):
        return self.loss_function(self.forward(q, p, t), phi)
    
    def loss_PDE(self, q_col, p_col, t_col):
        q = q_col.clone().detach()
        p = p_col.clone().detach()
        t = t_col.clone().detach()

        q.requires_grad = True
        p.requires_grad = True
        t.requires_grad = True

        f = self.forward(q, p, t)
        
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

        return self.loss_function(LHS, torch.zeros(f.shape).to(self.device))

    def compute_loss(self):
        q = self.X_train[:, 0].reshape(-1, 1)
        p = self.X_train[:, 1].reshape(-1, 1)
        t = self.X_train[:, 2].reshape(-1, 1)
        f = self.y_train.reshape(-1, 1)

        if not self.pinn:
            return self.loss_IC(q, p, t, f)
        else:
            q_col = self.X_train_with_col[:, 0].reshape(-1, 1)
            p_col = self.X_train_with_col[:, 1].reshape(-1, 1)
            t_col = self.X_train_with_col[:, 2].reshape(-1, 1)

            return (1 - self.weight) * self.loss_IC(q, p, t, f) + self.weight * self.loss_PDE(q_col, p_col, t_col)
    
    def closure(self):
        
        # reset gradients to zero:
        self.optimizer.zero_grad()
        
        # compute loss
        self.loss = self.compute_loss()
        
        # derivative with respect to model's weights:
        self.loss.backward()
        
        self.iter += 1
        
        if self.iter % 100 == 0:
            print(f"Iteration: {self.iter}, Loss = {self.loss:06f}")
            self.iter_list.append(int(self.iter+1))
            self.loss_list.append(float(self.loss))
        
        return self.loss
    
    def train(self):
        self.net.train()
        self.optimizer.step(self.closure)