from importlib import reload
import logging
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pformat
import os
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from skimage.metrics import structural_similarity as ssim 
from scipy.stats import pearsonr
from matplotlib.ticker import FuncFormatter


# Load configuration
with open("./setup_turb2d.yaml", "r") as f:
    setup = yaml.safe_load(f)

# Setup paths and device
sys.path.append(setup["repo_dir"])
output_dir = setup["output_dir"]
models_dir = setup["models_dir"]
data_dir = setup["data_dir"]
logging_dir = setup["logging_dir"]
device = setup["torch_device"]

if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)

# Import custom modules
import models
reload(models)
from models import simple_unet
from models import fno2d
reload(simple_unet)
reload(fno2d)
from models.simple_unet import SimpleUnet, SimpleUnetCond
from models.fno2d import FNO2D_grid, FNO2D_grid_tembedding, FNO2D_grid_tembedding_cond
from models import loss_functions
reload(loss_functions)
import utilities
reload(utilities)
from utilities import n2c, c2n, pthstr, linear_beta_scheduler, cosine_beta_scheduler


# Get current time for model naming
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")


# Load hyperparameters
with open("./ddpm_turb2d_config.yml", 'r') as h:
    hyperparam_dict = yaml.load(h, Loader=yaml.FullLoader)


# Extract hyperparameters
timesteps = hyperparam_dict["timesteps"]
beta_start = hyperparam_dict["beta_start"]
beta_end = hyperparam_dict["beta_end"]
batch_size = hyperparam_dict["batch_size"]
epochs = hyperparam_dict["epochs"]
loss_function = hyperparam_dict["loss_function"]
loss_function_start = hyperparam_dict["loss_function_start"]
loss_function_start_batch = hyperparam_dict["loss_function_start_batch"]
loss_args_start = hyperparam_dict["loss_args_start"]
loss_args_end = hyperparam_dict["loss_args_end"]
beta_scheduler = hyperparam_dict["beta_scheduler"]
ddpm_arch = hyperparam_dict["ddpm_arch"]
ddpm_params = hyperparam_dict["ddpm_params"]
train_type = hyperparam_dict["train_type"]
lr = hyperparam_dict["lr"]
data_type = hyperparam_dict["data_type"]
model_name = hyperparam_dict["model_name"]


 # Initialize noise scheduler
if beta_scheduler == "linear":
        betas, alphas, alphas_cumprod = linear_beta_scheduler(beta_start, beta_end, timesteps, device=device)
elif beta_scheduler == "cosine":
        betas, alphas, alphas_cumprod = cosine_beta_scheduler(timesteps, device=device)



DATA="99 Sparse_Turbulence"
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler(f"{logging_dir}/ddpm_qgm_losses_{current_time}.log"),
    logging.StreamHandler()
])
printlog = logging.info
printlog("-"*40)
printlog(f"Running test_final.py for {DATA} to Compare UNET, FNO, DDPM+UNET, DDPM+FNO")
printlog(f"loaded ddpm_config: {pformat(hyperparam_dict)}")
printlog("-"*40)

# Create model directory
model_dir = f"{models_dir}/{model_name}"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Data dimensions
Nx = 256
Ny = 256
numchannels = 1
lead = 0  



#############
## TESTING ##
#############
print("Starting  testing ...")

def min_max_denormalize(data, min_val, max_val):
    """
    Denormalize data scaled with Min-Max normalization to its original scale.

    Parameters:
        data (array): The normalized data.
        min_val (float): The minimum value of the original data before normalization.
        max_val (float): The maximum value of the original data before normalization.

    Returns:
        denormalized_data (array): The denormalized data.
    """
    if data.ndim == 4: 
        denorm = data[:, 0, :, :] * (max_val - min_val) + min_val
        denorm = denorm[:, np.newaxis, :, :]  
    elif data.ndim == 6:  
        denorm = data[:, :, :, 0, :, :] * (max_val - min_val) + min_val
        denorm = denorm[:, :, :, np.newaxis, :, :] 
    else:
        raise ValueError("Unsupported data shape: {}".format(data.shape))
    
    print("After Denorm shape", denorm.shape)
    return denorm


min_val = -47.3168934  
max_val =  48.1056251


# Setup for sampling and evaluation
seeds_total = 100
seeds_plot = 15
seeds_idx = np.arange(seeds_total)       
seeds_idx_plot = np.arange(seeds_plot)    

stausteps = 0
ftausteps = timesteps
timesteps_rev_list = np.arange(stausteps, ftausteps)
its = np.arange(len(timesteps_rev_list))
its_timesteps = np.array(list(zip(its, reversed(timesteps_rev_list))))
num_rev_steps = len(its_timesteps)

# Select timesteps for plotting
idx_skip = int(len(timesteps_rev_list)/40)
its_timesteps_plot_og = its_timesteps[::idx_skip]
its_timesteps_plot = np.concatenate((its_timesteps_plot_og, its_timesteps[-idx_skip:]), axis=0)
its_timesteps_plot2 = its_timesteps_plot_og
itrevfinal = np.where(its_timesteps[:,1] == 0)[0][0]


# Initialize the model architecture based on ddpm_arch
if ddpm_arch == "unet":
    model = SimpleUnet(**ddpm_params).to(device)
elif ddpm_arch == "unet_cond":
    model = SimpleUnetCond(**ddpm_params).to(device)
elif ddpm_arch == "fno2d":
    model = FNO2D_grid_tembedding_cond(**ddpm_params).to(device)


def gridgen(Lx, Ly, Nx, Ny, INDEXING='ij'):
    '''
    Generate a 2D grid.

    Parameters:
    -----------
    Lx : float
        Length of the domain in the x-direction.
    NX : int
        Number of grid points in the x and y-directions.
    INDEXING : str, optional
        Convention to use for indexing. Default is 'ij' (matrix indexing).

    Returns:
    --------
    Lx : float
        Length of the domain in the x-direction.
    Lx : float
        Length of the domain in the y-direction (same as x-direction as grid is square).
    X : numpy.ndarray
        2D array of x-coordinates.
    Y : numpy.ndarray
        2D array of y-coordinates.
    dx : float
        Size of grid spacing in the x-direction.
    dx : float
        Size of grid spacing in the y-direction (same as x-direction as grid is square).
    '''

    # Calculate the size of the grid spacing
    dx = Lx / Nx
    dy = Ly / Ny

    # Create an array of x-coordinates, ranging from 0 to (Lx - dx)
    x = np.linspace(0, Lx - dx, num=Nx)
    y = np.linspace(0, Lx - dx, num=Ny)

    # Create 2D arrays of the x and y-coordinates using a meshgrid.
    X, Y = np.meshgrid(x, y, indexing=INDEXING)

    # Return the lengths of the domain, the x and y-coordinates, and the size of the grid spacing.
    return Lx, Ly, X, Y, dx, dy


Lx = 2 * np.pi
NX = 256
_, _, X, Y, dx, dy = gridgen(Lx, Lx, NX, NX)



# Helper function to ensure consistent colorbar formatting
def format_colorbar_ticks(cb):
    cb.ax.tick_params(labelsize=16)
    for tick_label in cb.ax.get_yticklabels():
        tick_label.set_fontweight('bold')
        tick_label.set_fontsize(20)

# Generate comparison plots
for istep, s in enumerate(seeds_idx_plot, 0):

    print(f"Plotting timestep: {istep}")
    str_step = f"{istep:06d}"
    channels = ["vorticity"] 
    fixed_colorbars = {"vorticity":(-30, 30)}
    ch_name = channels[0]
    vmin, vmax = fixed_colorbars[ch_name]
    
    fig = plt.figure(figsize=(8, 20))  
    gs = gridspec.GridSpec(nrows=6, ncols=1, figure=fig)
    axs = [fig.add_subplot(gs[i, 0]) for i in range(6)]

    axs[0].pcolormesh(X, Y, test_input_sparse_use_np[s, 0].T, cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
    axs[0].set_title(f"Sparse Observations", fontsize=18, fontweight='bold')
    axs[0].set_aspect('equal') 
    axs[0].yaxis.set_ticks_position('both')     
    axs[0].tick_params(labelright=True) 
    axs[0].set_yticks([0, 2, 4, 6])  
    axs[0].set_xticks([0, 2, 4, 6])
    axs[0].tick_params(labelsize=16)
    axs[0].tick_params(axis='both', direction='out', length=6, width=1.5)

    for label in axs[0].get_xticklabels() + axs[0].get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(20)

    valid_ocean_coords = np.where(test_input_sparse_use_np[s, 0] != 0)
    axs[0].pcolormesh(X, Y, np.zeros_like(test_input_sparse_use_np[s, 0]).T, cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
    axs[0].scatter(X[valid_ocean_coords], Y[valid_ocean_coords], color='red', s=5)

    plot1 = axs[1].pcolormesh(X, Y, test_UNET_pred_sparse_denorm_np[s, 0].T, cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
    axs[1].set_title(f"UNET", fontsize=18, fontweight='bold')
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb1 = fig.colorbar(plot1, cax=cax, ticks=[vmin, 0, vmax], extend='both')
    format_colorbar_ticks(cb1)
    axs[1].set_aspect('equal') 

    plot2 = axs[2].pcolormesh(X, Y, test_FNO_pred_sparse_denorm_np[s, 0].T, cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
    axs[2].set_title(f"FNO", fontsize=18, fontweight='bold')
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb2 = fig.colorbar(plot2, cax=cax, ticks=[vmin, 0, vmax], extend='both')
    format_colorbar_ticks(cb2)
    axs[2].set_aspect('equal') 

    img3 = UNET_DDPM_probe_pred_denorm_np[s, 0, 0, 0, :, :]
    plot3 = axs[3].pcolormesh(X, Y, img3.T, cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
    axs[3].set_title(f"UNET+DDPM", fontsize=18, fontweight='bold')
    divider = make_axes_locatable(axs[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb3 = fig.colorbar(plot3, cax=cax, ticks=[vmin, 0, vmax], extend='both')
    format_colorbar_ticks(cb3)
    axs[3].set_aspect('equal') 

    img4 = FNO_DDPM_probe_pred_denorm_np[s, 0, 0, 0, :, :]
    plot4 = axs[4].pcolormesh(X, Y, img4.T, cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
    axs[4].set_title(f"FNO+DDPM", fontsize=18, fontweight='bold')
    divider = make_axes_locatable(axs[4])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb4 = fig.colorbar(plot4, cax=cax, ticks=[vmin, 0, vmax], extend='both')
    format_colorbar_ticks(cb4)
    axs[4].set_aspect('equal') 

    plot5 = axs[5].pcolormesh(X, Y, truth_test_denorm_np[s, 0].T, cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
    axs[5].set_title(f"Truth", fontsize=18, fontweight='bold')
    divider = make_axes_locatable(axs[5])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb5 = fig.colorbar(plot5, cax=cax, ticks=[vmin, 0, vmax], extend='both')
    format_colorbar_ticks(cb5)
    axs[5].set_aspect('equal') 

    for i, ax in enumerate(axs):
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
        if i != len(axs) - 1:
            ax.set_xticks([])
        else:
            ax.set_xticks([0, 2, 4, 6])
            ax.set_xlabel("X", fontsize=18, fontweight='bold')
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
                label.set_fontsize(20)
        ax.set_yticks([0, 2, 4, 6])
        ax.set_ylabel("Y", fontsize=18, fontweight='bold', rotation=0, labelpad=30, va='center')

        
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(20)

    plt.tight_layout()
    img_dir = f"{true_sparse_pred_diff}/true_sparse_pred_diff_timestep-{str_step}"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    plt.savefig(f"{img_dir}/channel-{ch_name}.png", dpi=300)
    plt.close()





# Helper function for FFT analysis
def compute_y_avg_fft(data):
    """Compute FFT along y-axis and average the magnitudes"""
    data_fft = np.fft.rfft(data, axis=1)
    return np.mean(np.abs(data_fft), axis=0)


# Prepare data for FFT analysis

arrays_to_plot = [
    test_input_sparse_use.cpu().numpy(),             
    test_FNO_pred_sparse_denorm.cpu().numpy(),           
    test_UNET_pred_sparse_denorm.cpu().numpy(),           
    FNO_DDPM_probe_pred_denorm[:, 0, 0, :, :, :].cpu().numpy(),  
    UNET_DDPM_probe_pred_denorm[:, 0, 0, :, :, :].cpu().numpy(), 
    truth_test_denorm.cpu().numpy(),                   
]


labels = {
    "Sparse Observations": "blue",
    "FNO": "red",
    "UNET": "orange",
    "FNO+DDPM": "green",
    "UNET+DDPM": "purple",
    "Truth": "black"
}

chs = ["vorticity"]
for i_ch, ch in enumerate(chs):
    fig, ax = plt.subplots(figsize=(4.2, 3.6), constrained_layout=True)
    for data_array, (model_label, color) in zip(arrays_to_plot, labels.items()):
        print(f"{model_label} shape: {data_array.shape}")
        fft_values = []
        for s in range(data_array.shape[0]):
            fft_s = compute_y_avg_fft(data_array[s, i_ch, :, :])
            fft_values.append(fft_s[2:-1])

        fft_values = np.stack(fft_values, axis=0)
        ks = np.arange(fft_values.shape[1]) + 2

        fft_mean = np.mean(fft_values, axis=0)
        fft_std = np.std(fft_values, axis=0)

        ax.plot(ks, fft_mean, label=model_label, color=color, linewidth=2)
        ax.fill_between(ks, fft_mean - fft_std, fft_mean + fft_std, color=color, alpha=0.3)

    ax.set_xlabel(r"$\mathbf{k}_{\mathbf{x}}$", fontsize=8)
    ax.set_ylabel(r"$\left|\boldsymbol{\hat{\zeta}}(\boldsymbol{k})\right|$", fontsize=8)

    for tick in ax.get_xticklabels():
        tick.set_fontsize(6)
        tick.set_fontweight('bold')

    for tick in ax.get_yticklabels():
        tick.set_fontsize(6)
        tick.set_fontweight('bold')

    ax.legend(fontsize=5, loc="upper right", bbox_to_anchor=(0.98, 0.98), ncol=1, frameon=True)
   

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)



############### RMSE ##################################################################################

def compute_rmse_per_sample(pred, truth):
    """
    Compute RMSE per sample.
    Inputs: [N, 1, H, W] or [N, H, W]
    Output: array of RMSE values, shape [N]
    """
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, truth {truth.shape}")
    
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
        truth = truth[:, 0]

    print("RMSE shape debug:", pred.shape, truth.shape)
    N = pred.shape[0]
    rmses = np.zeros(N)
    for i in range(N):
        diff = pred[i] - truth[i]
        rmses[i] = np.sqrt(np.mean(diff**2))
    return rmses

rmse_dict = {}
rmse_std_dict = {}

for name, model_pred in zip(["FNO", "UNET", "FNO+DDPM", "UNET+DDPM"],
                             [fno_pred, unet_pred, fno_ddpm_pred, unet_ddpm_pred]):
    rmses = compute_rmse_per_sample(model_pred, truth)
    rmse_dict[name] = np.mean(rmses)
    rmse_std_dict[name] = np.std(rmses)


labels = list(rmse_dict.keys())
means = [rmse_dict[l] for l in labels]
stds = [rmse_std_dict[l] for l in labels]

plt.figure(figsize=(4.2, 3.6))

bars = plt.bar(labels, means, yerr=stds, capsize=5, color='royalblue')
plt.ylabel(r"RMSE ($\pm \boldsymbol{\sigma}$)", fontsize=8, fontweight='bold')
plt.xticks(rotation=0, ha='center')  # Set rotation to 0 for horizontal labels
for label in plt.gca().get_xticklabels():
    label.set_fontsize(8)
    label.set_fontweight('bold')

plt.ylim(0, 7)  # Ensures the y-axis includes 7
plt.yticks(np.arange(0, 7.1, 1), fontsize=6, fontweight='bold')  # Show ticks from 0 to 7
plt.grid(axis='y', alpha=0.3)
# plt.tight_layout()

for bar in bars:
    bar.set_linewidth(0)
    bar.set_edgecolor("none")
    bar.set_path_effects([])

plt.savefig(f"{path_outputs_model_timesteps}/rmse_comparisonbar_with_error.png", dpi=300,  bbox_inches='tight')
plt.close()


####CC######################################################

def get_cc(preds, truths):
    """
    Calculate Pearson correlation coefficient between predictions and ground truth.
    
    Args:
        preds (np.ndarray): Predicted values with shape (steps, lat, lon, channels)
        truths (np.ndarray): Ground truth values with shape (steps, lat, lon, channels)
    
    Returns:
        np.ndarray: Correlation coefficients with shape (steps, channels)
    """

    assert preds.shape == truths.shape
    ccs = np.empty((preds.shape[0], preds.shape[3]))

    for istep in range(preds.shape[0]):
        for ich in range(preds.shape[-1]):
            ccs[istep,ich] = pearsonr(preds[istep,:,:,ich].flatten(),truths[istep,:,:,ich].flatten())[0]

    return ccs



# STEP1: Convert shape: [N, 1, H, W] -> [N, H, W, 1]
truth_cc = np.transpose(truth, (0, 2, 3, 1))
fno_cc = np.transpose(fno_pred, (0, 2, 3, 1))
unet_cc = np.transpose(unet_pred, (0, 2, 3, 1))
fno_ddpm_cc = np.transpose(fno_ddpm_pred, (0, 2, 3, 1))
unet_ddpm_cc = np.transpose(unet_ddpm_pred, (0, 2, 3, 1))

print("cc dimension", truth_cc.shape, fno_cc.shape, unet_cc.shape,  fno_ddpm_cc.shape, unet_ddpm_cc.shape)

assert not np.allclose(unet_ddpm_pred[:, 0], unet_pred[:, 0])  # Should not be identical

# STEP2: compute cc
cc_dict = {}
cc_std_dict = {}

for name, pred in zip(["FNO", "UNET", "FNO+DDPM", "UNET+DDPM"],
                      [fno_cc, unet_cc, fno_ddpm_cc, unet_ddpm_cc]):
    ccs = get_cc(pred, truth_cc)  # shape [N, C]
    cc_dict[name] = np.mean(ccs)  # mean over samples and channels
    cc_std_dict[name] = np.std(ccs)

labels = list(cc_dict.keys())
means = [cc_dict[k] for k in labels]
stds = [cc_std_dict[k] for k in labels]

# plt.figure(figsize=(6, 4))
plt.figure(figsize=(4.2, 3.6))
bars = plt.bar(labels, means, yerr=stds, capsize=5, color='royalblue')
# plt.ylabel("Correlation Coefficient (± Std)")
plt.ylabel(r"Correlation Coefficient ($\pm \boldsymbol{\sigma}$)",fontsize=8, fontweight='bold')
plt.ylim(0, 1)
plt.xticks(rotation=0, ha='center')  # Set rotation to 0 for horizontal labels
for label in plt.gca().get_xticklabels():
    label.set_fontsize(8)
    label.set_fontweight('bold')
plt.yticks(fontsize=6, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

for bar in bars:
    bar.set_linewidth(0)
    bar.set_edgecolor("none")

plt.savefig(f"{path_outputs_model_timesteps}/correlation_coefficient_with_error.png", dpi=300)
plt.close()


############### SSIM ##############################################################
def get_ssim(preds, truths):
    """
    Calculate Structural Similarity Index (SSIM) between predictions and ground truth.
    
    Args:
        preds (np.ndarray): Predicted values with shape (steps, lat, lon, channels)
        truths (np.ndarray): Ground truth values with shape (steps, lat, lon, channels)
    
    Returns:
        np.ndarray: SSIM values with shape (steps, channels)
    """
    assert preds.shape == truths.shape
    ssims = np.empty((truths.shape[0], truths.shape[3]))
    for step in range(truths.shape[0]):
        for ch in range(truths.shape[3]):
            ssims[step, ch] = ssim(
                preds[step, :, :, ch],
                truths[step, :, :, ch],
                data_range=truths[step, :, :, ch].max() - truths[step, :, :, ch].min()
            )
    return ssims


ssim_dict = {}
ssim_std_dict = {}

for name, pred in zip(["FNO", "UNET", "FNO+DDPM", "UNET+DDPM"],
                      [fno_cc, unet_cc, fno_ddpm_cc, unet_ddpm_cc]):
    ssims = get_ssim(pred, truth_cc)  # shape: [N, C]
    ssim_dict[name] = np.mean(ssims)
    ssim_std_dict[name] = np.std(ssims)

labels = list(ssim_dict.keys())
means = [ssim_dict[k] for k in labels]
stds = [ssim_std_dict[k] for k in labels]

plt.figure(figsize=(4.2, 3.6))
bars = plt.bar(labels, means, yerr=stds, capsize=5, color='royalblue')
# plt.ylabel("SSIM ± Std")
plt.ylabel(r"SSIM ($\pm \boldsymbol{\sigma}$)", fontsize=8, fontweight='bold')


plt.ylim(0, 0.6)
plt.xticks(rotation=0, ha='center') 
for label in plt.gca().get_xticklabels():
    label.set_fontsize(8)
    label.set_fontweight('bold')
plt.yticks(fontsize=6, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

for bar in bars:
    bar.set_linewidth(0)
    bar.set_edgecolor("none")

plt.savefig(f"{path_outputs_model_timesteps}/SSIM_with_error.png", dpi=300)
plt.close()
