import numpy as np
import torch
from scipy.linalg import inv

def scale_back_t(args, scaled_t: float):
    t = scaled_t / (args.w0 ** 2 / (args.k * args.T))
    return t

def scale_back(args, scaled_q: float, scaled_p: float, scaled_t: float):
    q = scaled_q / np.sqrt(args.m * args.w0 ** 2 / (args.k * args.T))
    p = scaled_p / np.sqrt(1 / (args.m * args.k * args.T))
    t = scale_back_t(args, scaled_t)

    return q, p, t

def scale_back_fermi(args, scaled_q: float, scaled_p: float, scaled_t: float):
    q = scaled_q * args.q_F
    p = scaled_p * args.p_F
    t = scaled_t / args.w0

    return q, p, t

def omega(args, scaled_t: float) -> float:
    if args.case_idx == 0:
        return 0.0
    elif args.case_idx == 1 or args.case_idx == 2:
        return args.w0
    elif args.case_idx == 3:
        if type(scaled_t) == torch.Tensor:
            return args.w0 * torch.sqrt(1 + scaled_t ** 2 / 10000)
        else:
            return args.w0 * np.sqrt(1 + scaled_t ** 2 / 10000)

def hamiltonian(args, scaled_q: float, scaled_p: float, scaled_t: float) -> float:

    if args.case_idx == 2:
        if scaled_t > 10:
            scaled_q = scaled_q + 0.5

    if args.fermi_scaling:
        q, p, t = scale_back_fermi(args, scaled_q, scaled_p, scaled_t)
    else:
        q, p, t = scale_back(args, scaled_q, scaled_p, scaled_t)

    w = omega(args, scaled_t)
    h = p ** 2 / (2 * args.m) + 0.5 * args.m * w ** 2 * q ** 2
    return h / (args.k * args.T)

def prob_den(args, scaled_q: float, scaled_p: float, scaled_t: float) -> float:
    if args.fermi_scaling:
        q, p, t = scale_back_fermi(args, scaled_q, scaled_p, scaled_t)
    else:
        q, p, t = scale_back(args, scaled_q, scaled_p, scaled_t)
        
    A = np.array([q, p])
    alpha = (1 + (args.w0 * t) ** 2) / args.w0 ** 2
    beta = args.m * t
    gamma = args.m ** 2
    
    Phi = (args.k * args.T / args.m) * np.array([[alpha, beta], [beta, gamma]])
    Phi_inv = inv(Phi)

    if args.fermi_scaling:
        #return args.N * args.N0 * np.exp(-0.5 * A.T @ Phi_inv @ A)
        return args.N * args.N0 * np.exp(-args.N * args.N0 * args.hbar * (2 * np.pi) * (q ** 2 + p ** 2))
    else:
        return args.N0 * np.exp(-0.5 * A.T @ Phi_inv @ A)