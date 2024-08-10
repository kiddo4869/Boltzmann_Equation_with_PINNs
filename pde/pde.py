import numpy as np
from scipy.linalg import inv

def scale_back(args, scaled_q: float, scaled_p: float, scaled_t: float):
    q = scaled_q / np.sqrt(args.m * args.w0 ** 2 / (args.k * args.T))
    p = scaled_p / np.sqrt(1 / (args.m * args.k * args.T))
    t = scaled_t / (args.w0 ** 2 / (args.k * args.T))

    return q, p, t

def scale_back_fermi(args, scaled_q: float, scaled_p: float, scaled_t: float):
    q = scaled_q * args.q_F
    p = scaled_p * args.p_F
    t = scaled_t / args.w0

    return q, p, t

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
        return args.N * args.N0 * np.exp(-args.N * args.N0 * args.hbar * (2 * np.pi) * (scaled_q ** 2 + scaled_p ** 2))
    else:
        return args.N0 * np.exp(-0.5 * A.T @ Phi_inv @ A)