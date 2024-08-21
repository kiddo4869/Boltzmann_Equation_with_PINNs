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
    q_arr = np.linspace(q_min, q_max, args.grid_size, dtype=np.float64)
    p_arr = np.linspace(p_min, p_max, args.grid_size, dtype=np.float64)
    t_arr = np.linspace(t_min, t_max, args.grid_size, dtype=np.float64)
    pv, qv, tv = np.meshgrid(p_arr, q_arr, t_arr, indexing="ij")
    
    # Exact solution
    f_exact = np.array([[prob_den(args, q, p, 0.0) for q in q_arr] for p in p_arr])
    plot_init_solution(args, q_arr, p_arr, f_exact)

    path = os.path.join(args.checkpoint, "state_dict_model.pth")

    if args.phase == "train":

        # Data processing
        train_inputs, train_init_pts, train_init_phi, train_col_pts = data_processing_with_time(qv, pv, tv, f_exact,
                                                                                                args.N_boundary, args.N_collocation,
                                                                                                ub, lb, t_range)
        val_inputs, val_init_pts, val_init_phi, val_col_pts = data_processing_with_time(qv, pv, tv, f_exact,
                                                                                        args.N_boundary_val, args.N_collocation_val,
                                                                                        ub, lb, t_range)

        # Plotting data inputs
        plot_initial_collocation_points(args, train_init_pts, train_col_pts, "Training Initial and Collocation Points")
        plot_initial_collocation_points(args, val_init_pts, val_col_pts, "Validation Initial and Collocation Points")

        # Data conversion
        X_train = torch.from_numpy(train_init_pts).float().to(args.device)
        y_train = torch.from_numpy(train_init_phi).float().to(args.device)
        X_train_with_col = torch.from_numpy(train_inputs).float().to(args.device)

        X_valid = torch.from_numpy(val_init_pts).float().to(args.device)
        y_valid = torch.from_numpy(val_init_phi).float().to(args.device)
        X_valid_with_col = torch.from_numpy(val_inputs).float().to(args.device)

        X_flatten = torch.from_numpy(qv.reshape(-1, 1)).to(args.device)
        Y_flatten = torch.from_numpy(pv.reshape(-1, 1)).to(args.device)
        T_flatten = torch.from_numpy(tv.reshape(-1, 1)).to(args.device)

        f_exact_gpu = torch.from_numpy(f_exact).to(args.device)

        # Print shapes
        log_shapes(X_train, y_train, X_train_with_col, X_valid, y_valid, X_valid_with_col, X_flatten, Y_flatten, T_flatten)

        # Model creation
        args.pinn = True
        model = PINN(args, X_train, y_train, X_train_with_col,
                           X_valid, y_valid, X_valid_with_col)
        model.to(args.device)

        # Print number of parameters
        params = list(model.parameters())
        num_of_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

        logging.info(f"Number of parameters: {num_of_params}")
        logging.info(f"Input shape: {X_train.shape}")

        # Trial before training
        trial_test(X_train, y_train, X_train_with_col, model)

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
        model = PINN(args, torch.empty(0), torch.empty(0), torch.empty(0),
                           torch.empty(0), torch.empty(0), torch.empty(0))
        model.net.load_state_dict(torch.load(path))
        model.net.eval()

        print("start testing...")
        logging.info("start testing...")
        testing_start = time.perf_counter()

        # plotting solutions and distributions
        sol_files = []
        dis_files = []

        spacing = (q_max - q_min) / args.grid_size

        if args.debug:  
            t_arr = np.linspace(0, 50, 6)
        else:
            t_arr = np.linspace(0, 100, 21)
        for i, t in enumerate(t_arr):
            if i % args.ds_freq == 0:# and i != 0:
                if args.dynamic_scaling:
                    # method 1: multiply original grid by a factor of ds_ratio (different spacing)
                    #q_arr = q_arr * args.ds_ratio
                    # method 2: extend the grid with the same spacing
                    # (q_max-q_min)/args.grid_size: original spacing
                    #q_arr = np.arange(q_arr[0]*args.ds_ratio, q_arr[-1]*args.ds_ratio, (q_max-q_min)/args.grid_size)
                    # method 3: extend the grid with the same spacing
                    q_arr = np.arange(q_arr[0]-spacing*args.ds_grid_add, q_arr[-1]+spacing*(args.ds_grid_add+1), spacing)

            sol_files.append(plot_solution(args, q_arr, p_arr, t, prob_den, model))
            dis_files.append(plot_q_p_distributions(args, q_arr, p_arr, t, model))

        if args.dynamic_scaling:
            save_gif_PIL(os.path.join(args.checkpoint, "solutions_ds.gif"), sol_files, fps=5, loop=0)
            save_gif_PIL(os.path.join(args.checkpoint, "q_p_distributions_ds.gif"), dis_files, fps=5, loop=0)
        else:
            save_gif_PIL(os.path.join(args.checkpoint, "solutions.gif"), sol_files, fps=5, loop=0)
            save_gif_PIL(os.path.join(args.checkpoint, "q_p_distributions.gif"), dis_files, fps=5, loop=0)

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

def log_shapes(X_train, y_train, X_train_with_col, X_valid, y_valid, X_valid_with_col, X_flatten, Y_flatten, T_flatten):
    logging.info(f"X_train shape: {X_train.shape}")
    logging.info(f"y_train shape: {y_train.shape}")
    logging.info(f"X_train_with_cal shape: {X_train_with_col.shape}\n")

    logging.info(f"X_val shape: {X_valid.shape}")
    logging.info(f"y_val shape: {y_valid.shape}")
    logging.info(f"X_val_with_cal shape: {X_valid_with_col.shape}\n")

    logging.info(f"X_flatten shape: {X_flatten.shape}")
    logging.info(f"Y_flatten shape: {Y_flatten.shape}")
    logging.info(f"T_flatten shape: {T_flatten.shape}")

def trial_test(X_train, y_train, X_train_with_col, model):
    q_trial = X_train[:, 0].reshape(-1, 1)
    p_trial = X_train[:, 1].reshape(-1, 1)
    t_trial = X_train[:, 2].reshape(-1, 1)

    q_trial_with_col = X_train_with_col[:, 0].reshape(-1, 1)
    p_trial_with_col = X_train_with_col[:, 1].reshape(-1, 1)
    t_trial_with_col = X_train_with_col[:, 2].reshape(-1, 1)

    print(f"q_trial shape: {q_trial.shape}")
    print(f"p_trial shape: {p_trial.shape}")
    print(f"t_trial shape: {t_trial.shape}")

    f_trial = model(q_trial, p_trial, t_trial)

    print(f"f_trial shape: {f_trial.shape}")
    print(f"y_train shape: {y_train.shape}")

    # IC loss
    IC_loss = model.loss_IC(q_trial, p_trial, t_trial, y_train.reshape(-1, 1)).item()
    print(f"IC loss: {IC_loss}")

    # PDE loss
    PDE_loss = model.loss_PDE(q_trial_with_col, p_trial_with_col, t_trial_with_col).item()
    print(f"PDE loss: {PDE_loss}")

    # sum of the loss with weight
    print(f"sum of the loss with weight: {(1 - model.PDE_weight) * IC_loss + model.PDE_weight * PDE_loss}")

    # total loss
    total_loss = model.compute_loss()[0].item()
    print(f"total loss: {total_loss}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Physics Informed Neural Networks")
    # Physical Conditions
    parser.add_argument("--grid_size", type=int, default=100)
    parser.add_argument("--T", type=float, default=1.7e-07)
    parser.add_argument("--w0", type=float, default=1e5)
    parser.add_argument("--k", type=float, default=sc.physical_constants['Boltzmann constant in Hz/K'][0])
    parser.add_argument("--m", type=float, default=sc.m_u * 86.9091835)
    parser.add_argument("--fermi_scaling", action="store_true")

    # Data Sampling
    parser.add_argument("--q_min_max", type=json.loads, default=[-5, 5])
    parser.add_argument("--p_min_max", type=json.loads, default=[-5, 5])
    parser.add_argument("--t_min_max", type=json.loads, default=[0, 50])
    parser.add_argument("--N_boundary", type=int, default=200, help="Number of training data")
    parser.add_argument("--N_collocation", type=int, default=600, help="Number of training collocation points")
    parser.add_argument("--N_boundary_val", type=int, default=100, help="Number of validation data")
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
    parser.add_argument("--layers", type=json.loads, default=[3,20,20,20,20,1])
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument("--hamiltonian", action="store_true")

    # Training parameters
    parser.add_argument("--PDE_weight", type=float, default=0.25)
    parser.add_argument("--optimizer", type=str, default="l-bfgs", help="adam or l-bfgs")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-5)

    # Visualization
    parser.add_argument("--dynamic_scaling", action="store_true")
    parser.add_argument("--ds_freq", type=int, default=1)
    parser.add_argument("--ds_ratio", type=float, default=1.05)
    parser.add_argument("--ds_grid_add", type=int, default=3)

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

    if args.hamiltonian:
        args.layers[-1] = 2

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
            os.path.join(args.checkpoint, "solutions_ds")])
    
    main(args)
