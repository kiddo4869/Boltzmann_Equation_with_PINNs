import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import trapz
import torch
import torch.nn as nn
from PIL import Image
from data.data_processing import create_inputs, find_hamiltonian_seperately

def plot_data_inputs(args, data_list, title):
    if args.hamiltonian == "output":
        sampled_initial_points, sampled_interior_points, collocation_points = data_list
    else:
        sampled_initial_points, collocation_points = data_list

    plt.close()

    # Creating figure
    fig = plt.figure(figsize = (9, 9))
    ax = plt.axes(projection ="3d")

    # Creating plot
    ax.scatter3D(sampled_initial_points[:, 0], sampled_initial_points[:, 1], sampled_initial_points[:, 2], color="blue", label="Initial Points")
    ax.scatter3D(collocation_points[:, 0], collocation_points[:, 1], collocation_points[:, 2], color="red", label="Collocation Points")

    if args.hamiltonian == "output":
        ax.scatter3D(sampled_interior_points[:, 0], sampled_interior_points[:, 1], sampled_interior_points[:, 2], color="green", label="Interior Points")
    
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_zlabel("t")
    ax.view_init(elev=25, azim=-65)
    
    plt.legend()
    plt.title(title)
    
    path = os.path.join(args.checkpoint, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(path)

def plot_loss(ax, iter_list, train_loss_list, valid_loss_list, title_text):
    ax.set_title(title_text)
    ax.set_xlabel("epoch")
    ax.set_ylabel("losses")
    ax.plot(iter_list, train_loss_list, label="training loss")
    ax.plot(iter_list, valid_loss_list, label="validation loss")
    ax.legend()

def plot_losses(save_to_dir: str, model):

    plt.close()
    fig = plt.figure(1, figsize=(18, 12))
    plt.suptitle(f"losses against epoch", fontsize=16)

    # IC Loss
    ax1 = plt.subplot(2, 2, 1)
    plot_loss(ax1, model.iter_list, model.train_ic_ll, model.valid_ic_ll, "Initial Condition Loss")

    # BC Loss
    ax2 = plt.subplot(2, 2, 2)
    plot_loss(ax2, model.iter_list, model.train_bc_ll, model.valid_bc_ll, "Boundary Condition Loss")
    
    # PDE Loss
    ax2 = plt.subplot(2, 2, 3)
    plot_loss(ax2, model.iter_list, model.train_pde_ll, model.valid_pde_ll, "PDE Loss")
    
    # Total Loss
    ax3 = plt.subplot(2, 2, 4)
    plot_loss(ax3, model.iter_list, model.train_loss_list, model.valid_loss_list, "Total Loss")

    # tight layout
    plt.tight_layout()
    
    path = os.path.join(save_to_dir, f"losses.png")
    plt.savefig(path)

def plot_data(ax, q_arr, p_arr, data, title_text, plot_hamiltonian=False, set_aspect=False):
    im = ax.contourf(q_arr, p_arr, data, levels=100)
    ax.set_title(title_text)
    ax.set_xlabel("q' (no unit)")
    ax.set_ylabel("p' (no unit)")
    if set_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0,0))
    label = "Hamiltonian" if plot_hamiltonian else "Probability Density"
    plt.colorbar(im, ax=ax, label=label)

def plot_init_solution(args,
                       scaled_q_arr: np.array,
                       scaled_p_arr: np.array,
                       exact_solution: np.array,
                       title_text,
                       plot_hamiltonian=False,
                       set_aspect=False):

    plt.close()
    fig = plt.figure(1, figsize=(8, 8))
    ax = plt.subplot(1, 1, 1)
    if plot_hamiltonian:
        plot_data(ax, scaled_q_arr, scaled_p_arr, exact_solution, title_text, plot_hamiltonian, set_aspect)
    else:
        plot_data(ax, scaled_q_arr, scaled_p_arr, exact_solution, title_text, set_aspect=set_aspect)
    path = os.path.join(args.checkpoint, f"{title_text.replace(' ', '_').lower()}.png")
    plt.savefig(path)
    plt.tight_layout()

    return path

def plot_solution(args,
                  scaled_q_arr: np.array,
                  scaled_p_arr: np.array,
                  scaled_t: float,
                  func,
                  model,
                  output):
    plt.close()
    
    fig = plt.figure(1, figsize=(20, 6))

    plt.suptitle(f"Solution at t' = {scaled_t}", fontsize=16)
    
    # Ground truth
    ax1 = plt.subplot(1, 3, 1)
    f_exact = np.array([[func(args, scaled_q, scaled_p, scaled_t) for scaled_q in scaled_q_arr] for scaled_p in scaled_p_arr])
    plot_data(ax1, scaled_q_arr, scaled_p_arr, f_exact, "Exact Solution")

    q_trial, p_trial, t_trial = create_inputs(scaled_q_arr, scaled_p_arr, scaled_t)
    h_trial = torch.from_numpy(find_hamiltonian_seperately(args, q_trial, p_trial, t_trial)).to(args.device)
    
    N_q = len(scaled_q_arr)
    N_p = len(scaled_p_arr)
    
    # Prediction
    ax2 = plt.subplot(1, 3, 2)
    if args.hamiltonian == "output":
        if output == "f":
            f_pred = model(q_trial, p_trial, t_trial, h_trial)[:, 0].reshape(N_p, N_q).cpu().detach().numpy()
        elif output == "h":
            f_pred = model(q_trial, p_trial, t_trial, h_trial)[:, 1].reshape(N_p, N_q).cpu().detach().numpy()
        else:
            raise ValueError(f"Invalid output type: {output}")
    else:
        f_pred = model(q_trial, p_trial, t_trial, h_trial).reshape(N_p, N_q).cpu().detach().numpy()
        
    plot_data(ax2, scaled_q_arr, scaled_p_arr, f_pred, "Prediction")
    
    # Error
    ax3 = plt.subplot(1, 3, 3)
    plot_data(ax3, scaled_q_arr, scaled_p_arr, np.abs(f_exact - f_pred), r"Absolute Error $|\phi(x, y) - \hat \phi(x, y)|$")
    # add MSE value under the plot
    mse_loss_instance = nn.MSELoss()
    mse_loss = mse_loss_instance(torch.from_numpy(f_exact), torch.from_numpy(f_pred)).item()
    ax3.text(-2, -7, f"MSE = {mse_loss:05f}", fontsize=13)

    if output == "f":
        folder = "solutions"
    elif output == "h":
        folder = "hamiltonian"
    else:
        raise ValueError(f"Invalid output type: {output}")

    if args.dynamic_scaling:
        path = os.path.join(args.checkpoint, folder + "_ds", f"solution_at_t={scaled_t}.png")
    else:
        path = os.path.join(args.checkpoint, folder, f"solution_at_t={scaled_t}.png")

    plt.savefig(path)

    print(f"Saved solution at t' = {scaled_t} to {path}")

    return path

def plot_q_p_distributions(args, scaled_q_arr, scaled_p_arr, scaled_t, model):

    plt.close()

    q_trial, p_trial, t_trial = create_inputs(scaled_q_arr, scaled_p_arr, scaled_t)
    h_trial = torch.from_numpy(find_hamiltonian_seperately(args, q_trial, p_trial, t_trial)).to(args.device)
    
    N_q = len(scaled_q_arr)
    N_p = len(scaled_p_arr)
    if args.hamiltonian == "output":
        f_distrib = model(q_trial, p_trial, t_trial, h_trial)[:, 0].reshape(N_p, N_q).cpu().detach().numpy()
    else:
        f_distrib = model(q_trial, p_trial, t_trial, h_trial).reshape(N_p, N_q).cpu().detach().numpy()

    # Simplified scaling of q_arr and p_arr
    q_arr = scaled_q_arr / np.sqrt(args.m * args.w0**2 / (args.k * args.T))
    p_arr = scaled_p_arr / np.sqrt(1 / (args.m * args.k * args.T))
    t = scaled_t / (args.w0**2 / (args.k * args.T))

    # Directly compute momentum and position distributions using vectorized operations
    momentum_distribution = trapz(f_distrib, q_arr)
    position_distribution = trapz(f_distrib.T, p_arr)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    ax = axs[1, 1]
    plt.contourf(q_arr, p_arr, f_distrib, levels=100)
    ax.set_title(f"Prediction at t = {t * 1000: 02f} (ms)")
    ax.set_xlabel("q (m)")
    ax.set_ylabel("p (kg ms^-1)")
    plt.colorbar(label="Probability Density")
    plt.ticklabel_format(style="sci", axis="both", scilimits=(0,0))

    # Second subplot: Distribution of q
    ax = axs[0, 1]
    ax.plot(q_arr, position_distribution)
    ax.axvline(x=0, color="red", linestyle='--')
    ax.set_title('Distribution of Position q')

    # Third subplot: Distribution of p
    ax = axs[1, 0]
    ax.plot(p_arr, momentum_distribution)
    ax.axvline(x=0, color="red", linestyle='--')
    ax.set_title('Distribution of Momentum p')

    # Hide the top-right subplot
    axs[0, 0].set_axis_off()

    # prevent the subplots moving over animations
    plt.tight_layout()

    if args.dynamic_scaling:
        path = os.path.join(args.checkpoint, "q_p_distributions_ds", f"q_p_distributions_at_t={scaled_t}.png")
    else:
        path = os.path.join(args.checkpoint, "q_p_distributions", f"q_p_distributions_at_t={scaled_t}.png")
    plt.savefig(path)

    print(f"Saved q and p distributions at t' = {scaled_t} to {path}")

    return path

def save_gif_PIL(outfile, files, fps=10, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)