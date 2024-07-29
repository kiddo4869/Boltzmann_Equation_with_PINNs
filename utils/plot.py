import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from data.data_processing import create_inputs

def plot_initial_collocation_points(args, sampled_initial_points, collocation_points, title):
    plt.close()

    # Creating figure
    fig = plt.figure(figsize = (9, 9))
    ax = plt.axes(projection ="3d")

    # Creating plot
    ax.scatter3D(sampled_initial_points[:, 0], sampled_initial_points[:, 1], sampled_initial_points[:, 2], color="blue", label="Initial Points")
    ax.scatter3D(collocation_points[:, 0], collocation_points[:, 1], collocation_points[:, 2], color="red", label="Collocation Points")
    
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_zlabel("t")
    ax.view_init(elev=25, azim=-65)
    
    plt.legend()
    plt.title(title)
    
    path = os.path.join(args.checkpoint, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(path)
    
    # show plot
    #plt.show()

def plot_data(ax, q_arr, p_arr, data, title_text):
    im = ax.pcolor(q_arr, p_arr, data)
    ax.set_title(title_text)
    ax.set_xlabel("q' (no unit)")
    ax.set_ylabel("p' (no unit)")
    ax.set_aspect("equal", adjustable="box")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0,0))
    plt.colorbar(im, ax=ax, label="Probability Density")

def plot_losses(save_to_dir: str, iter_list: list, loss_list: list):    
    plt.close()
    plt.plot(iter_list, loss_list, label="training loss")
    #plt.plot(iter_list, loss_list, label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("losses")
    plt.title("losses against epoch graph")
    plt.legend()
    
    path = os.path.join(save_to_dir, f"losses.png")
    plt.savefig(path)
    #plt.show()

def plot_solution(args,
                  scaled_q_arr: np.array,
                  scaled_p_arr: np.array,
                  scaled_t: float,
                  func,
                  model):
    plt.close()
    
    fig = plt.figure(1, figsize=(20, 6))
    
    # Ground truth
    ax1 = plt.subplot(1, 3, 1)
    f_exact = np.array([[func(args, scaled_q, scaled_p, scaled_t) for scaled_q in scaled_q_arr] for scaled_p in scaled_p_arr])
    plot_data(ax1, scaled_q_arr, scaled_p_arr, f_exact, "Exact Solution")

    q_trial, p_trial, t_trial = create_inputs(scaled_q_arr, scaled_p_arr, scaled_t)
    
    N_q = len(scaled_q_arr)
    N_p = len(scaled_p_arr)
    
    # Prediction
    ax2 = plt.subplot(1, 3, 2)
    f_pred = model(q_trial, p_trial, t_trial).reshape(N_q, N_p).cpu().detach().numpy()
    plot_data(ax2, scaled_q_arr, scaled_p_arr, f_pred, "Prediction")
    
    # Error
    ax3 = plt.subplot(1, 3, 3)
    plot_data(ax3, scaled_q_arr, scaled_p_arr, np.abs(f_exact - f_pred), r"Absolute Error $|\phi(x, y) - \hat \phi(x, y)|$")
    # add MSE value under the plot
    mse_loss_instance = nn.MSELoss()
    mse_loss = mse_loss_instance(torch.from_numpy(f_exact), torch.from_numpy(f_pred)).item()
    ax3.text(-2, -7, f"MSE = {mse_loss:05f}", fontsize=13)
    
    path = os.path.join(args.checkpoint, f"evluation_at_t={scaled_t}.png")
    plt.savefig(path)
    
    #plt.show()

def save_gif_PIL(outfile, files, fps=10, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)