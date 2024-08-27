from typing import List, Optional, Tuple, Dict, Union
import numpy as np
import torch
from pyDOE import lhs

def stack_data(input_list: list, phi_data: np.array):
    inputs = np.hstack([input_data.reshape(-1, 1) for input_data in input_list])
    phi_data = phi_data.reshape(-1, 1)
    return inputs, phi_data

def select_collocation_points(N_collocation: int,
                              ub: np.array,
                              lb: np.array,
                              t_range: np.array) -> np.array:

    collocation_time = t_range[0] + (t_range[1] - t_range[0]) * lhs(2, N_collocation)[:, 0].reshape(-1, 1)
    collocation_space = lb + (ub - lb) * lhs(2, N_collocation)
    collocation_points = np.concatenate([collocation_space, collocation_time], axis=1)

    return collocation_points # shape: (N_collocation, 3)

def get_initial_points(list_to_stack: list[np.array]) -> np.array:
    # Initial condition at t=0 or its min value
    initial_points = np.column_stack([array[:, :, 0].reshape(-1) for array in list_to_stack])
    return initial_points

def get_boundary_points(list_to_stack: list[np.array]) -> np.array:
    up = np.column_stack([array[0, :, :].reshape(-1) for array in list_to_stack])
    down = np.column_stack([array[-1, :, :].reshape(-1) for array in list_to_stack])
    left = np.column_stack([array[:, 0, :].reshape(-1) for array in list_to_stack])
    right = np.column_stack([array[:, -1, :].reshape(-1) for array in list_to_stack])

    return np.vstack((up, down, left, right))

def select_points(idx: np.array, points: np.array) -> np.array:
    sampled_points = points[idx, :].copy()
    return sampled_points

def preprocess_data(args,
                    X: np.array,
                    Y: np.array,
                    T: np.array,
                    phi: np.array,
                    ham: np.array,
                    ub: np.array,
                    lb: np.array,
                    t_range: np.array) -> Tuple[np.array, np.array, np.array, np.array]:

    # Get initial points
    initial_points = get_initial_points([X, Y, T])
    initial_phi = phi.reshape(-1, 1)
    initial_ham = ham.reshape(-1, 1)

    # Sample random points in initial condition
    idx = np.random.choice(initial_points.shape[0], args.N_initial, replace=False)
    sampled_initial_points = select_points(idx, initial_points)
    sampled_initial_phi = select_points(idx, initial_phi)
    sampled_initial_ham = select_points(idx, initial_ham)

    # Get boundary points
    boundary_points = get_boundary_points([X, Y, T])
    
    # Sample random points in boundary condition
    idx = np.random.choice(boundary_points.shape[0], args.N_boundary, replace=False)
    sampled_boundary_points = select_points(idx, boundary_points)
    #sampled_boundary_ham = select_points(idx, initial_ham)
    
    # Sample collocation points in the domain
    collocation_points = select_collocation_points(args.N_collocation, ub, lb, t_range)

    if not args.hamiltonian:
        return sampled_initial_points, sampled_initial_phi, sampled_boundary_points, collocation_points
    else:
        return sampled_initial_points, sampled_initial_phi, sampled_initial_ham, sampled_boundary_points, collocation_points

def create_inputs(q_arr: np.array, p_arr: np.array, t: float):
    pv, qv = np.meshgrid(p_arr, q_arr, indexing="ij")
    coor_pts = np.column_stack((qv.reshape(-1), pv.reshape(-1)))
    q_trial = torch.from_numpy(coor_pts[:, 0].reshape(-1, 1))
    p_trial = torch.from_numpy(coor_pts[:, 1].reshape(-1, 1))
    t_trial = t * torch.ones(q_trial.shape[0]).reshape(-1, 1)
    
    return q_trial, p_trial, t_trial

def convert_data(args, data: np.array):
    return torch.from_numpy(data).float().to(args.device)