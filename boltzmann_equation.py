import os
import json
import argparse
from typing import List, Optional, Tuple, Dict, Union
import logging
import time
from datetime import datetime
import random

import numpy as np
from numpy.linalg import inv
import scipy.constants as sc
import torch

from data.data_processing import *
from utils.util import *
from utils.plot import *
from models.network import *
from pde.pde import *

def main(args: argparse.ArgumentParser):
    random.seed(args.seed)
    np.random.seed(args.seed)
    start = time.perf_counter()

    initialize_logging(args)
    
    args_text = write_args(args)
    with open(os.path.join(args.checkpoint, "args.txt"), "w") as file:
        file.write(args_text)

    log_parameters(args)

    # Initialize data
    print("initialize data points...")
    logging.info("initialize data points...")

    q_min, q_max = args.q_min_max
    p_min, p_max = args.p_min_max
    t_min, t_max = args.t_min_max

    # Bounds and ranges
    ub = np.array([q_max, p_max])
    lb = np.array([q_min, p_min])
    t_range = np.array([t_min, t_max])

    # Grids
    q_arr = np.linspace(q_min, q_max, args.N_q, dtype=np.float32)
    p_arr = np.linspace(p_min, p_max, args.N_p, dtype=np.float32)
    t_arr = np.linspace(t_min, t_max, args.N_t, dtype=np.float32)
    pv, qv, tv = np.meshgrid(p_arr, q_arr, t_arr, indexing="ij")
    print(f"pv shape: {pv.shape}")
    print(f"qv shape: {qv.shape}")
    print(f"tv shape: {tv.shape}")

    # Exact solution
    if args.case_idx == 2:
        f_exact = np.array([[prob_den(args, q, p, t_arr[0]) for q in q_arr] for p in p_arr])  
    else:
        f_exact = np.array([[prob_den(args, q, p, t_arr[0]) for q in q_arr] for p in p_arr])
    
    #h_exact_over_time = np.array([[[hamiltonian(args, q, p, t) for q in q_arr] for p in p_arr] for t in t_arr])
    #h_exact_over_time = np.transpose(h_exact_over_time, (1, 2, 0))
    
    plot_init_solution(args, q_arr, p_arr, f_exact, "Initial Probability Density", set_aspect=True)
    h_exact = np.array([[hamiltonian(args, q, p, 0) for q in q_arr] for p in p_arr])
    plot_init_solution(args, q_arr, p_arr, h_exact, "Initial Hamiltonian H", plot_hamiltonian=True, set_aspect=True)

    """
    for i in range(len(t_arr)):
        h_exact = np.array([[hamiltonian(args, q, p, t_arr[i]) for q in q_arr] for p in p_arr])
        plot_init_solution(args, q_arr, p_arr, h_exact, f"Hamiltonian H' {i}", plot_hamiltonian=True)
    """

    path = os.path.join(args.checkpoint, "state_dict_model.pth")

    # Data processing
    train_inputs, train_labels = preprocess_data(args, qv, pv, tv, f_exact,
                                                 ub, lb, t_range)

    if args.hamiltonian == "input":
        args.h_min_max = [train_inputs[0][:, 3].min(), train_inputs[0][:, 3].max()]
        print(f"hamiltonian min: {args.h_min_max[0]}, max: {args.h_min_max[1]}")

    if args.phase == "train":

        val_inputs, val_labels = preprocess_data(args, qv, pv, tv, f_exact,
                                                 ub, lb, t_range, valid=True)

        # Plotting data inputs
        plot_data_inputs(args, train_inputs, "Training Inputs")
        plot_data_inputs(args, val_inputs, "Validation Inputs")

        # Data conversion
        train_inputs = convert_data(args, train_inputs)
        train_labels = convert_data(args, train_labels)
        val_inputs = convert_data(args, val_inputs)
        val_labels = convert_data(args, val_labels)

        # Model creation
        args.pinn = True
        model = PINN(args, train_inputs, train_labels,
                           val_inputs, val_labels)
        model.to(args.device)

        # Print number of parameters
        params = list(model.parameters())
        num_of_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

        logging.info(f"Number of parameters: {num_of_params}")

        # Trial before training
        #trial_test(args, train_inputs, train_labels, model)

        print("start training...")
        logging.info("start training...")

        training_start = time.perf_counter()

        if args.optimizer == "adam":
            model.train_with_adam()
        elif args.optimizer == "l-bfgs":
            model.train_with_lbfgs()
        
        training_end = time.perf_counter()
        print(f"training time elapsed: {(training_end - training_start):02f}s")
        logging.info(f"training time elapsed: {(training_end - training_start):02f}s")
    
        #plotting losses
        if args.log_loss:
            plot_losses(args.checkpoint, model)
    
        # saving model
        torch.save(model.net.state_dict(), path)
    elif args.phase == "test":
        print("load saved model...")
        logging.info("load saved model...")

        # loading model
        train_inputs = [torch.empty(0) for _ in range(3 if args.hamiltonian == "output" else 2)]
        train_labels = [torch.empty(0) for _ in range(2 if args.hamiltonian == "output" else 1)]
        val_inputs = [torch.empty(0) for _ in range(3 if args.hamiltonian == "output" else 2)]
        val_labels = [torch.empty(0) for _ in range(2 if args.hamiltonian == "output" else 1)]
        model = PINN(args, train_inputs, train_labels, val_inputs, val_labels)
        model.net.load_state_dict(torch.load(path))
        model.net.eval()
        
        """
        q = torch.tensor([float("inf")]).reshape(-1, 1)
        p = torch.tensor([0.0]).reshape(-1, 1)
        t = torch.tensor([0.0]).reshape(-1, 1)
        h = hamiltonian(args, q, p, t).reshape(-1, 1)
        test_result = model(q, p, t, h)
        print(test_result)

        exit()
        """
        print("start testing...")
        logging.info("start testing...")
        testing_start = time.perf_counter()

        # plotting solutions and distributions
        sol_files = []
        dis_files = []
        ham_files = []

        spacing = (q_max - q_min) / args.grid_size

        if args.debug:  
            t_arr = np.linspace(0, 100, 6)
        else:
            t_arr = np.linspace(0, 100, 21)
        for i, t in enumerate(t_arr):
            if i % args.ds_freq == 0:# and i != 0:
                if args.dynamic_scaling:
                    q_arr = np.arange(q_arr[0]-spacing*args.ds_grid_add, q_arr[-1]+spacing*(args.ds_grid_add+1), spacing)

            sol_files.append(plot_solution(args, q_arr, p_arr, t, prob_den, model, output="f"))
            dis_files.append(plot_q_p_distributions(args, q_arr, p_arr, t, model))
            if args.hamiltonian == "output":
                ham_files.append(plot_solution(args, q_arr, p_arr, t, hamiltonian, model, output="h"))

        if args.dynamic_scaling:
            save_gif_PIL(os.path.join(args.checkpoint, "solutions_ds.gif"), sol_files, fps=5, loop=0)
            save_gif_PIL(os.path.join(args.checkpoint, "q_p_distributions_ds.gif"), dis_files, fps=5, loop=0)
            if args.hamiltonian == "output":
                save_gif_PIL(os.path.join(args.checkpoint, "hamiltonian_ds.gif"), ham_files, fps=5, loop=0)
        else:
            save_gif_PIL(os.path.join(args.checkpoint, "solutions.gif"), sol_files, fps=5, loop=0)
            save_gif_PIL(os.path.join(args.checkpoint, "q_p_distributions.gif"), dis_files, fps=5, loop=0)
            if args.hamiltonian == "output":
                save_gif_PIL(os.path.join(args.checkpoint, "hamiltonian.gif"), ham_files, fps=5, loop=0)

        testing_end = time.perf_counter()
        print(f"testing time elapsed: {(testing_end - testing_start):02f}s")
        logging.info(f"testing time elapsed: {(testing_end - testing_start):02f}s")
    else:
        pass

    end = time.perf_counter()
    print(f"program time elapsed: {(end - start):02f}s")
    logging.info(f"program time elapsed: {(end - start):02f}s")
    

def initialize_logging(args):
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(args.log_path, f"{current_date}.log")
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

def log_parameters(args):
    string = "\n--------------Physical Parameters--------------\n"
    string += f"\nTemperature: {args.T} K"
    string += f"\nTrapping Frequency: {args.w0} Hz"
    string += f"\nBoltzmann constant: {args.k} Hz/K"
    string += f"\nMass of the atom: {args.m} kg"
    string += f"\nNormalization Constant: {args.N0} (no unit)"

    if args.fermi_scaling:
        string += f"\nN: {args.N} (no unit)"
        string += f"\nhbar: {args.hbar} kg m^2 s^-2"
        string += f"\nq_F: {args.q_F} m"
        string += f"\np_F: {args.p_F} kg m s^-1"
    
    logging.info(string)

def log_shapes(train_ic_pts, train_init_phi, train_col_pts, val_ic_pts, val_init_phi, val_col_pts, X_flatten, Y_flatten, T_flatten):
    logging.info(f"train_ic_pts shape: {train_ic_pts.shape}")
    logging.info(f"train_init_phi shape: {train_init_phi.shape}")
    logging.info(f"train_col_pts shape: {train_col_pts.shape}\n")

    logging.info(f"val_ic_pts shape: {val_ic_pts.shape}")
    logging.info(f"val_init_phi shape: {val_init_phi.shape}")
    logging.info(f"val_col_pts shape: {val_col_pts.shape}\n")

    logging.info(f"X_flatten shape: {X_flatten.shape}")
    logging.info(f"Y_flatten shape: {Y_flatten.shape}")
    logging.info(f"T_flatten shape: {T_flatten.shape}")

def compute_loss(model, inputs, labels, loss_type="IC"):

    q = inputs[:, 0].reshape(-1, 1)
    p = inputs[:, 1].reshape(-1, 1)
    t = inputs[:, 2].reshape(-1, 1)
    
    print("q shape: ", q.shape)
    print("p shape: ", p.shape)
    print("t shape: ", t.shape)
    if args.hamiltonian == "input":
        h = inputs[:, 3].reshape(-1, 1)
        print("h shape: ", h.shape)
    print("labels shape: ", labels.shape)

    if loss_type == "IC":
        loss = model.loss_IC(q, p, t, h, labels).item()
    elif loss_type == "BC":
        loss = model.loss_BC(q, p, t, h, labels).item()
    elif loss_type == "PDE":
        loss = model.loss_PDE(q, p, t, h).item()
    else:
        raise ValueError("Invalid loss type")
    
    print(f"{loss_type} loss: {loss}")
    return loss

def trial_test(args, inputs_list, labels_list, model):

    IC_loss = compute_loss(model, inputs_list[0], labels_list[0], "IC")
    BC_loss = compute_loss(model, inputs_list[1], labels_list[1], "BC") if args.hamiltonian == "output" else 0.0
    PDE_loss = compute_loss(model, inputs_list[-1], np.empty(0), "PDE")

    # sum of the loss with weight
    sum_loss = model.IC_weight * IC_loss + model.BC_weight * BC_loss + model.PDE_weight * PDE_loss

    # total loss
    total_loss = model.compute_loss()[0].item()

    print(f"sum of the loss with weight: {sum_loss}, type: {type(sum_loss)}")
    print(f"total loss: {total_loss}, type: {type(total_loss)}")
    print(f"diff: {sum_loss - total_loss}, type: {type(sum_loss - total_loss)}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Physics Informed Neural Networks")
    # Physical Conditions
    parser.add_argument("--grid_size", type=int, default=100)
    parser.add_argument("--N_q", type=int, default=100)
    parser.add_argument("--N_p", type=int, default=100)
    parser.add_argument("--N_t", type=int, default=100)
    parser.add_argument("--T", type=float, default=1.7e-07)
    parser.add_argument("--w0", type=float, default=1e5)
    parser.add_argument("--k", type=float, default=sc.physical_constants['Boltzmann constant in Hz/K'][0])
    parser.add_argument("--m", type=float, default=sc.m_u * 86.9091835)
    parser.add_argument("--fermi_scaling", action="store_true")

    # Data Sampling
    parser.add_argument("--q_min_max", type=json.loads, default=[-5, 5])
    parser.add_argument("--p_min_max", type=json.loads, default=[-5, 5])
    parser.add_argument("--t_min_max", type=json.loads, default=[0, 50])
    parser.add_argument("--N_initial", type=int, default=200, help="Number of training data")
    parser.add_argument("--N_boundary", type=int, default=100, help="Number of training boundary points")
    parser.add_argument("--N_collocation", type=int, default=600, help="Number of training collocation points")
    parser.add_argument("--N_initial_val", type=int, default=100, help="Number of validation data")
    parser.add_argument("--N_boundary_val", type=int, default=100, help="Number of validation boundary points")
    parser.add_argument("--N_collocation_val", type=int, default=300, help="Number of validation collocation points")

    # General parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="base")
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--log_path", type=str, default="./logs")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pinn", action="store_true")
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--log_loss", action="store_true")
    parser.add_argument("--log_sol", action="store_true")

    # PINNs parameters
    parser.add_argument("--layers", type=json.loads, default=[3,10,20,50,80,50,20,10,1])
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument("--hamiltonian", type=str, default=None)

    # Training parameters
    parser.add_argument("--IC_weight", type=float, default=1.0)
    parser.add_argument("--BC_weight", type=float, default=1.0)
    parser.add_argument("--PDE_weight", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="l-bfgs", help="adam or l-bfgs")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--history_size", type=int, default=100)
    parser.add_argument("--tolerance_grad", type=float, default=1e-05)
    parser.add_argument("--tolerance_change", type=float, default=1e-09)

    # Visualization
    parser.add_argument("--dynamic_scaling", action="store_true")
    parser.add_argument("--ds_freq", type=int, default=1)
    parser.add_argument("--ds_ratio", type=float, default=1.05)
    parser.add_argument("--ds_grid_add", type=int, default=3)

    parser.add_argument("--case_idx", type=int, default=0)

    args = parser.parse_args()

    if args.debug:
        args.epochs = 100
        args.log_loss = True
        args.log_freq = 10

    if args.fermi_scaling:
        args.N=10e5
        args.hbar=1
        #args.k=1
        args.T=1e-6
        args.w0=3800
        args.m=sc.m_u*6.04
        args.q_min_max=[-1, 1]
        args.p_min_max=[-1, 1]

    if args.hamiltonian == "input":
        args.layers[0] += 1
    elif args.hamiltonian == "output":
        args.layers[-1] += 1

    # Device setup
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", args.device)

    # Normalization Constant
    args.N0 = args.w0 / (2 * np.pi * args.k * args.T)
    if args.fermi_scaling:
        args.q_F = np.sqrt(2 * args.N * args.hbar / (args.m * args.w0))
        args.p_F = np.sqrt(2 * args.N * args.hbar * (args.m * args.w0))

    # Model Checkpoint
    args.checkpoint = os.path.join(args.checkpoint_path, args.name)
    mkdirs([args.checkpoint,
            os.path.join(args.checkpoint, "q_p_distributions"),
            os.path.join(args.checkpoint, "q_p_distributions_ds"),
            os.path.join(args.checkpoint, "solutions"),
            os.path.join(args.checkpoint, "solutions_ds"),
            os.path.join(args.checkpoint, "hamiltonian"),])
    
    main(args)
