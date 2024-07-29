from typing import List, Optional, Tuple, Dict, Union
import numpy as np
import torch
from pyDOE import lhs

def stack_data(input_list: list, Phi_data: np.array):
    inputs = np.hstack([input_data.reshape(-1, 1) for input_data in input_list])
    Phi_data = Phi_data.reshape(-1, 1)
    return inputs, Phi_data
    
def data_processing_with_time(X: np.array,
                              Y: np.array,
                              T: np.array,
                              phi: np.array,
                              N_boundary: int,
                              N_collocation: int,
                              ub: np.array,
                              lb: np.array,
                              t_range: np.array) -> Tuple[np.array, np.array, np.array, np.array]:
    
    # Initial condition at t=0 or its min value
    initial_condition_index = np.argmin(T) if len(T) > 0 else 0    
    initial_points = np.column_stack((X[:, :, initial_condition_index].reshape(-1),
                                      Y[:, :, initial_condition_index].reshape(-1),
                                      T[:, :, initial_condition_index].reshape(-1)))
    
    initial_phi = phi.reshape(-1)
    
    print(initial_points.shape, initial_phi.shape)

    # Sample random points in the boundary
    idx = np.random.choice(initial_points.shape[0], N_boundary, replace=False)
    
    sampled_initial_points = initial_points[idx, :]
    #print(sampled_initial_points)
    sampled_initial_phi = initial_phi[idx]
    
    print(sampled_initial_points.shape, sampled_initial_phi.shape)
    
    # Sample collocation points in the domain
    collocation_time = t_range[0] + (t_range[1] - t_range[0]) * lhs(2, N_collocation)[:, 0].reshape(-1, 1)
    collocation_space = lb + (ub - lb) * lhs(2, N_collocation)
    
    collocation_points = np.concatenate([collocation_space, collocation_time], axis=1)

    train_inputs = np.vstack((sampled_initial_points, collocation_points))

    return train_inputs, sampled_initial_points, sampled_initial_phi, collocation_points

def create_inputs(q_arr: np.array, p_arr: np.array, t: float):
    pv, qv = np.meshgrid(p_arr, q_arr, indexing="ij")
    coor_pts = np.column_stack((qv.reshape(-1), pv.reshape(-1)))
    q_trial = torch.from_numpy(coor_pts[:, 0].reshape(-1, 1))
    p_trial = torch.from_numpy(coor_pts[:, 1].reshape(-1, 1))
    t_trial = t * torch.ones(q_trial.shape[0]).reshape(-1, 1)
    
    return q_trial, p_trial, t_trial