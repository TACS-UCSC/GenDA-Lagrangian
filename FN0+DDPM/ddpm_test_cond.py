from importlib import reload
import logging
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
from datetime import datetime
from scipy.stats import norm
from pprint import pformat
import os

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
from models import Unet2d                
reload(Unet2d)
reload(simple_unet)
reload(fno2d)
from models.simple_unet import SimpleUnet, SimpleUnetCond
from models.fno2d import FNO2D_grid, FNO2D_grid_tembedding, FNO2D_grid_tembedding_cond
from models import loss_functions      
reload(loss_functions)
import utilities
reload(utilities)
from utilities import n2c, c2n, pthstr, linear_beta_scheduler, cosine_beta_scheduler
import metrics
reload(metrics)
from metrics import rfft_abs_mirror_torch

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


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler(f"{logging_dir}/ddpm_qgm_losses_{current_time}.log"),
    logging.StreamHandler()
])
printlog = logging.info
printlog("-"*40)
printlog(f"Running test for {model_name}...")
printlog(f"loaded ddpm_turb2d_config: {pformat(hyperparam_dict)}")
printlog("-"*40)

# Create model directory
model_dir = f"{models_dir}/{model_name}"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Data dimensions
Nx = 256
Ny = 256
numchannels = 1
lead = 0  # same time predictions


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
    if data.ndim == 4:  # (batch_size, num_channels, Nx, Ny)
        denorm = data[:, 0, :, :] * (max_val - min_val) + min_val
        denorm = denorm[:, np.newaxis, :, :]  # Add back channel dimension
    elif data.ndim == 6:  # (seeds, timesteps, 1, num_channels, Nx, Ny)
        denorm = data[:, :, :, 0, :, :] * (max_val - min_val) + min_val
        denorm = denorm[:, :, :, np.newaxis, :, :]  # Add back channel dimension
    else:
        raise ValueError("Unsupported data shape: {}".format(data.shape))
    
    print("After Denorm shape", denorm.shape)
    return denorm


min_val = -47.3168934  
max_val =  48.1056251



# Setup for sampling and evaluation
seeds = 100  # Number of samples to generate
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

# Define the directory where models are saved

# Initialize the model architecture based on ddpm_arch
if ddpm_arch == "unet":
    model = SimpleUnet(**ddpm_params).to(device)
elif ddpm_arch == "unet_cond":
    model = SimpleUnetCond(**ddpm_params).to(device)
elif ddpm_arch == "fno2d":
    model = FNO2D_grid_tembedding_cond(**ddpm_params).to(device)


# Prepare for inference
model.eval()

# Initializes two tensors to store:


probe_data = torch.empty((seeds, len(timesteps_rev_list), 1, numchannels, Nx, Ny), device="cpu")
probe_noise = torch.empty((seeds, len(timesteps_rev_list), 1, numchannels, Nx, Ny), device="cpu")
seeds_idx = np.arange(seeds)

# # Load test data

test_input_loc = "/glade/derecho/scratch/nasefi/compressed_sensing/Beta_channel-turbulence_n/save_models/dnMSEFNO99S_EP500/test_input_sparse.pkl"
test_pred_loc = "/glade/derecho/scratch/nasefi/compressed_sensing/Beta_channel-turbulence_n/save_models/dnMSEFNO99S_EP500/test_pred_sparse.pkl"
truth_test_loc = "/glade/derecho/scratch/nasefi/compressed_sensing/Beta_channel-turbulence_n/save_models/dnMSEFNO99S_EP500/test_truth.pkl"


with open(test_input_loc, "rb") as f:
    test_input_sparse = pickle.load(f)

with open(test_pred_loc, "rb") as f:
    test_pred_sparse = pickle.load(f)

with open(truth_test_loc, "rb") as f:
    truth_test = pickle.load(f)



test_pred_sparse = test_pred_sparse[:, :, :, :]

print("CheckDimFirst_Test", test_input_sparse.shape, test_pred_sparse.shape, truth_test.shape)


#Permute Channel for FNO inputs 
test_input_sparse = test_input_sparse.permute((0, 3, 1, 2))
test_pred_sparse = test_pred_sparse.permute((0, 3, 1, 2))
truth_test = truth_test.permute((0, 3, 1, 2))


data_test = test_pred_sparse
cond_data_test = test_pred_sparse
chunk_size = 2

# Run inference
print("Starting inference loop...")
with torch.no_grad():
    for start in range(0, seeds, chunk_size):
        end = min(start + chunk_size, seeds)
        for s in range(start, end):
                
            printlog(f"Seed: {s+1}/{seeds}")
            # Add noise to test data
            noise = torch.randn_like(data_test[[s]])
            noisy_data = torch.sqrt(alphas_cumprod[timesteps-1].view(-1, 1, 1, 1)) * data_test[[s]] + \
                        torch.sqrt(1 - alphas_cumprod[timesteps-1].view(-1, 1, 1, 1)) * noise
            x = noisy_data

            # Reverse diffusion process
            for it, t in its_timesteps:
                print(f"Inference on seed {s}, timestep {t}")
                timestep = torch.tensor([t], device=device)
                predicted_noise = model(x, cond_data_test[[s]], timestep)
                alpha_t = alphas[t]
                alpha_bar_t = alphas_cumprod[t]
                x = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise)

                # Add noise if using linear scheduler and not at final step
                if beta_scheduler == "linear" and t > stausteps:
                    teff = max(1, t)
                    beta_t = betas[teff]
                    x = x + torch.sqrt(beta_t) * torch.randn_like(x)

                # # Store results
                # probe_data[s, it] = x
                # probe_noise[s, it] = predicted_noise

                # After computing x and predicted_noise on GPU
                probe_data[s, it] = x.detach().cpu()
                probe_noise[s, it] = predicted_noise.detach().cpu()

print("probe_data bf", probe_data.shape)
final_step_data = probe_data[:, -1:, :, :, :, :]  # [seeds, 1, 1, channels, Nx, Ny]
print("final_step_data shape", final_step_data.shape)
probe_pred_data =  probe_data.permute(0, 1, 2, 3, 5, 4)
probe_pred_data =  probe_data

# print("probe_pred_data after", probe_pred_data.shape)
# probe_pred_noise = probe_noise.permute(0, 1, 2, 3, 5, 4)

train_input_sparse_use = test_input_sparse[seeds_idx].permute(0, 1, 3, 2)

train_pred_sparse_use = test_pred_sparse[seeds_idx].permute(0, 1, 3, 2)
train_truth_use = truth_test[seeds_idx].permute(0, 1, 3, 2)



# Save probe_pred_data
save_path_probe = "/glade/derecho/scratch/nasefi/compressed_sensing/Beta_channel-turbulence_n/save_models/save_probe/turblence2d/fno_ddpm_100seed/FNO_DDPM_probe_highepoch_turbu.pkl"
with open(save_path_probe, "wb") as f:
    pickle.dump(final_step_data, f)

print(f"Saved FNO+DDPM probe_pred_data to {save_path_probe}")

#denormalization 
min_val = -47.3168934  
max_val =  48.1056251

try:
    print("Running denormalization...")
    train_pred_sparse_denorm = min_max_denormalize(train_pred_sparse_use.cpu(),  min_val, max_val)
    print("train_pred_sparse_denorm", train_pred_sparse_denorm.shape)

    train_truth_denorm = min_max_denormalize(train_truth_use.cpu(), min_val, max_val)
    print("train_truth_denorm", train_truth_denorm.shape)

    probe_pred_data_denorm = min_max_denormalize(probe_pred_data.cpu(), min_val, max_val)
    print("probe_pred_data_denorm", probe_pred_data_denorm.shape)
    print("Denormalization complete.")
except Exception as e:
    print(f"Error during denormalization: {e}")


# Create output directories
path_outputs_model = f"{output_dir}/{model_name}"
if not os.path.exists(path_outputs_model):
    printlog(f"Creating directory: {path_outputs_model}")
    os.makedirs(path_outputs_model)
    
path_outputs_model_timesteps = f"{path_outputs_model}/ddpm_timesteps"
if not os.path.exists(path_outputs_model_timesteps):
    printlog(f"Creating directory: {path_outputs_model_timesteps}")
    os.makedirs(path_outputs_model_timesteps)


chs = ["vorticity"] 
yscales = ["linear", "log"]

true_sparse_pred_diff = f"{path_outputs_model_timesteps}/true_sparse_pred_diff"
if not os.path.exists(true_sparse_pred_diff):
    os.makedirs(true_sparse_pred_diff)
chs = ["vorticity"] 
true_sparse_pred_diff = f"{path_outputs_model_timesteps}/true_sparse_pred_diff"
os.makedirs(true_sparse_pred_diff, exist_ok=True)

# Get timesteps for visualization
itau1, tau1 = its_timesteps[0]     # τ = 999 
itau2, tau2 = its_timesteps[-10]   # τ = 9
itau3, tau3 = its_timesteps[-1]    # τ = 0  


# Generate comparison plots
for istep, s in enumerate(seeds_idx, 0):
    print(f"Plotting timestep: {istep}")
    str_step = f"{istep:06d}"
    channels = ["vorticity"] 

    #Convert Tensor to Numpy # I should copy the numpy otherwise it gets update the rest of data to nans 

    train_input_sparse_use_np = train_input_sparse_use.cpu().numpy().copy()
    train_pred_sparse_denorm_np = train_pred_sparse_denorm.cpu().numpy().copy()
    train_truth_denorm_np = train_truth_denorm.cpu().numpy().copy()
    probe_pred_data_denorm_np = probe_pred_data_denorm.cpu().numpy().copy()
    
    fixed_colorbars = {"vorticity":(-50, 50)}
    ch_name = channels[0]
    fig, axs = plt.subplots(1, 6, figsize=(18, 5))

    vmin, vmax = fixed_colorbars[ch_name]

    axs[0].imshow(train_input_sparse_use_np[s, 0, :, :], cmap='coolwarm',  vmin=vmin, vmax=vmax)
    axs[0].set_title(f"99% Sparse Input/ {ch_name}")
    valid_ocean_coords = np.where((train_input_sparse_use_np[s, 0, :, :] != 0)) 
    axs[0].scatter(valid_ocean_coords[1], valid_ocean_coords[0], color='red', s=5)  # Mark non-zero ocean points

    pred_plot= axs[1].imshow(train_pred_sparse_denorm_np[s, 0, :, :], cmap='coolwarm',  vmin=vmin, vmax=vmax)
    axs[1].set_title(f"FNO / {ch_name}")
    fig.colorbar(pred_plot, ax=axs[1], ticks=[vmin, 0, vmax], fraction=0.046, pad=0.04)

    probe_plotau1= axs[2].imshow(probe_pred_data_denorm_np[s, 0, 0, 0, :, :], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[2].set_title(r"FNO+DDPM $\tau$ = %i" % tau1)
    fig.colorbar(probe_plotau1, ax=axs[2], ticks=[vmin, 0, vmax], fraction=0.046, pad=0.04)

    probe_plotau2= axs[3].imshow(probe_pred_data_denorm_np[s, -10, 0, 0, :, :], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[3].set_title(r"FNO+DDPM $\tau$ = %i" % tau2)
    fig.colorbar(probe_plotau2, ax=axs[3], ticks=[vmin, 0, vmax], fraction=0.046, pad=0.04)

    probe_plotau3= axs[4].imshow(probe_pred_data_denorm_np[s, -1, 0, 0, :, :], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[4].set_title(r"FNO+DDPM $\tau$ = %i" % tau3)
    fig.colorbar(probe_plotau3, ax=axs[4], ticks=[vmin, 0, vmax], fraction=0.046, pad=0.04)

    truth_plot= axs[5].imshow(train_truth_denorm_np[s, 0, :, :], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[5].set_title(f"Truth /{ch_name}")
    fig.colorbar(truth_plot, ax=axs[5], ticks=[vmin, 0, vmax], fraction=0.046, pad=0.04)

    plt.suptitle(f"Comparison for Seed {s}\n full tau, conditioning/{timesteps}")
    plt.tight_layout()

    img_dir = f"{true_sparse_pred_diff}/true_sparse_pred_diff_timestep-{str_step}"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    plt.savefig(f"{img_dir}/channel-{ch_name}.png", dpi=300)
    plt.close()

# Create list of image paths for video creation
png_locs_txt = ""
for istep, s in enumerate(seeds_idx, 0):
    str_step = f"{istep:06d}"
    png_loc = f"{true_sparse_pred_diff}/true_sparse_pred_diff_timestep-{str_step}.png"
    png_locs_txt += f"{png_loc}\n"

true_sparse_pred_diff_txt_loc = f"{path_outputs_model_timesteps}/true_sparse_pred_diff_timestep.txt"    
with open(true_sparse_pred_diff_txt_loc, "w") as f:
    f.write(png_locs_txt)

# Create video from images
os.system(f'ffmpeg -y -r 10 -f image2 -s 1920x1080 -i {true_sparse_pred_diff}/true_sparse_pred_diff_timestep-%06d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {path_outputs_model_timesteps}/true_sparse_pred_diff.mp4')

# Generate simplified comparison plots (FNO, DDPM, Truth)
for istep, s in enumerate(seeds_idx, 0):
    print(f"Plotting timestep: {istep}")
    str_step = f"{istep:06d}"

    channels = ["vorticity"] 

    fixed_colorbars = {"vorticity":(-50, 50)}

    ch_name = channels[0]
    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    vmin, vmax = fixed_colorbars[ch_name]

    axs[0].imshow(train_input_sparse_use_np[s, 0, :, :], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[0].set_title(f"Input 99% Sparse / {ch_name}")
    valid_ocean_coords = np.where((train_input_sparse_use_np[s, 0, :, :] != 0))  # non-zero
    axs[0].scatter(valid_ocean_coords[1], valid_ocean_coords[0], color='red', s=5)  # Mark non-zero ocean points

    pred_plot = axs[1].imshow(train_pred_sparse_denorm_np[s, 0, :, :], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[1].set_title(f"FNO / {ch_name}")
    fig.colorbar(pred_plot, ax=axs[1], ticks=[vmin, 0, vmax], fraction=0.046, pad=0.04)
    
    prob_plot = axs[2].imshow(probe_pred_data_denorm_np[s, -1, 0, 0, :, :], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[2].set_title(r"FNO+DDPM $\tau$ = %i" % tau3)
    fig.colorbar(prob_plot, ax=axs[2], ticks=[vmin, 0, vmax], fraction=0.046, pad=0.04)

    truth_plot = axs[3].imshow(train_truth_denorm_np[s, 0, :, :], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[3].set_title(f"Truth/ {ch_name}")
    fig.colorbar(truth_plot, ax=axs[3], ticks=[vmin, 0, vmax], fraction=0.046, pad=0.04)

    plt.suptitle(f"Comparison for Seed {s} \n full tau, conditioning/{timesteps}")
    plt.tight_layout()

    img_dir = f"{true_sparse_pred_diff}/true_pred_timestep-{str_step}"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    plt.savefig(f"{img_dir}/channel-{ch_name}.png", dpi=300)
    plt.close()

# Create video from simplified comparison
os.system(f'ffmpeg -y -r 10 -f image2 -s 1920x1080 -i {true_sparse_pred_diff}/true_sparse_pred_diff_timestep-%06d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {path_outputs_model_timesteps}/true_pred_timestep.mp4')

# Helper function for FFT analysis
def compute_y_avg_fft(data):
    """Compute FFT along y-axis and average the magnitudes"""
    data_fft = np.fft.rfft(data, axis=1)
    return np.mean(np.abs(data_fft), axis=0)


# Prepare data for FFT analysis
#I just denormalize the fno pred, ddpm pred, truth. 
arrays_to_plot = [
    train_input_sparse_use.cpu().numpy(),
    train_pred_sparse_denorm.cpu().numpy(),
    probe_pred_data_denorm[:, 0, 0].cpu().numpy(),  #the t= 999
    probe_pred_data_denorm[:, -10, 0].cpu().numpy(), # the t= 9
    probe_pred_data_denorm[:, -1, 0].cpu().numpy(),   #final step t=0
    train_truth_denorm.cpu().numpy()
]



labels = {
    "99% Sparse Input": "blue",
    "FNO": "red",
    r"FNO+DDPM $\tau$ = %i" % tau1: "orange",
    r"FNO+DDPM $\tau$ = %i" % tau2: "purple",
    r"FNO+DDPM $\tau$ = %i" % tau3: "green",
    "Truth": "black"
}


# Plot FFT comparison
chs = ["vorticity"] 
for i_ch, ch in enumerate(chs):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot individual FFTs with transparency
    for data_array, (label, color) in zip(arrays_to_plot, labels.items()):
        for s in range(data_array.shape[0]):
            y_avg_fft = compute_y_avg_fft(data_array[s, i_ch, :, :])
            ks = np.arange(y_avg_fft.shape[0])
            ax.plot(ks[2:-1], y_avg_fft[2:-1], color=color, alpha=0.8, linewidth=1.5)

    # Add legend entries
    for _, (label, color) in zip(arrays_to_plot, labels.items()):
        ax.plot([], [], label=label, color=color, alpha=1)

    ax.set_xlabel('k')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Vertically Average Spectrum - {ch}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()

    if not os.path.exists(path_outputs_model_timesteps):
        os.makedirs(path_outputs_model_timesteps)
    plt.savefig(f"{path_outputs_model_timesteps}/y_avg_fft_comparison_{ch}.png", dpi=300)
    plt.close()


# Generate timestep visualization for selected seeds
for s in seeds_idx[:2]:  # Only use first two seeds for visualization
    plot_dir = f"{path_outputs_model_timesteps}/s-{s}"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for it, t in its_timesteps_plot:

        chs = ["vorticity"] 
        for i, ch in enumerate(chs):
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            
            # Get data limits for colorbar
            data = probe_data[s, it, 0, i]
            vmin = torch.min(data).cpu().numpy()
            vmax = torch.max(data).cpu().numpy()
            im = ax.imshow(data.cpu().numpy(), cmap='coolwarm')
            ax.set_title(f"{ch}")

            cbar = fig.colorbar(im, orientation='vertical', ticks=[vmin, 0, vmax], fraction=0.046, pad=0.04)
            

            save_path = f"{plot_dir}/probe_revStep_s-{s}_t-{pthstr(t)}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.suptitle(f"{model_name}\nrev timestep: {t} and {i}")
            plt.savefig(f"{save_path}/chs-{i}.png", dpi=300)
            plt.close()

# Spectrum plots of the probe_data
for revfinal in [len(its_timesteps)]:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    startk = 2

    # Calculate FFT of true data
    true_data_fft = rfft_abs_mirror_torch(train_truth_use, axis=3).mean(axis=0)
    true_data_fft_mean = torch.mean(true_data_fft, dim=1)

    ks = np.arange(true_data_fft_mean.shape[1])
    ax.plot([], [], color="black", linewidth=2, label="actual")
    ax.plot([], [], color="black", linewidth=2, label="true final reverse", linestyle="--")
    
    # Plot FFT for each timestep
    for it, t in its_timesteps_plot:
        # Calculate FFT of probe data at this timestep
        probe_data_fft = rfft_abs_mirror_torch(probe_data[:, it, 0], axis=3).mean(axis=0)
        probe_data_fft_mean = torch.mean(probe_data_fft, dim=1)

        # Add legend entry for selected timesteps
        if (it % (len(its_timesteps)//3)) == 0:
            ax.plot([], [], color=cm.rainbow(it/its_timesteps.shape[0]), label=f"reverse t: {t}")

        # Plot FFT
        ax.plot(ks[startk:], c2n(probe_data_fft_mean[0, startk:]), 
                color=cm.rainbow(it/its_timesteps.shape[0]), alpha=0.5)

    # Add colorbar for timesteps
    cbar = fig.colorbar(cm.ScalarMappable(cmap='rainbow'), ax=ax)
    numticks = 11
    cbar.set_ticks(np.linspace(0, 1, num=numticks))
    cbar.set_ticklabels([f"{int(t)}" for t in np.linspace(ftausteps, stausteps, num=numticks)])
    cbar.set_label('Reverse timestep')
    
    # Plot true data FFT
    ax.plot(ks[startk:], c2n(true_data_fft_mean[0, startk:]), color="black", linewidth=3)

    # Plot final reverse timestep FFT
    probe_data_fft = rfft_abs_mirror_torch(probe_data[:, itrevfinal, 0], axis=3).mean(axis=0)
    probe_data_fft_mean = torch.mean(probe_data_fft, dim=1)
    ax.plot(ks[startk:], c2n(probe_data_fft_mean[0, startk:]), color="black", linewidth=3, linestyle="--")

    ax.set_title(f"{chs[0]}")
    ax.set_xlabel('k')
    ax.set_ylabel('Amplitude')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.suptitle(f"Probe data FFT\nReverse timesteps from {ftausteps} to {stausteps}")
    plt.tight_layout()
    
    # Save with different y-scales
    for yscale in yscales:
        ax.set_yscale(yscale)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{path_outputs_model}/spectrum_revFinal-{revfinal}_yscale-{yscale}.png", dpi=200)

    plt.close()

# Spectrum difference plots
yscales_diff = ["linear", "symlog"]
for revfinal in [len(its_timesteps)]:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    startk = 2

    # Calculate FFT of true data
    true_data_fft = rfft_abs_mirror_torch(train_truth_use, axis=3).mean(axis=0).to(device)
    true_data_fft_mean = torch.mean(true_data_fft, dim=1).to(device)

    ks = np.arange(true_data_fft_mean.shape[1])
    
    # Plot FFT difference for each timestep
    for it, t in its_timesteps_plot:
        probe_data_fft = rfft_abs_mirror_torch(probe_data[:, it, 0], axis=3).mean(axis=0)
        probe_data_fft_mean = torch.mean(probe_data_fft, dim=1).to(device)
        
        # Plot difference between probe and true data FFT
        ax.plot(ks[startk:], c2n(probe_data_fft_mean[0, startk:] - true_data_fft_mean[0, startk:]), 
                color=cm.rainbow(it/its_timesteps.shape[0]), alpha=0.5)
    
    # Plot final timestep difference
    probe_data_fft = rfft_abs_mirror_torch(probe_data[:, itrevfinal, 0], axis=3).mean(axis=0)
    probe_data_fft_mean = torch.mean(probe_data_fft, dim=1).to(device)
    ax.plot(ks[startk:], c2n(probe_data_fft_mean[0, startk:] - true_data_fft_mean[0, startk:]), 
            color="black", linewidth=3, linestyle="--")

    # Add colorbar for timesteps
    cbar = fig.colorbar(cm.ScalarMappable(cmap='rainbow'), ax=ax)
    numticks = 11
    cbar.set_ticks(np.linspace(0, 1, num=numticks))
    cbar.set_ticklabels([f"{int(t)}" for t in np.linspace(ftausteps, stausteps, num=numticks)])
    cbar.set_label('Reverse timestep')

    ax.set_title(f"{chs[0]}")
    ax.set_xlabel('k')
    ax.set_ylabel('Amplitude')
    ax.grid(alpha=0.3)

    plt.suptitle(f"Probe data FFT, difference from true\nReverse timesteps from {ftausteps} to {stausteps}")

    # Save with different y-scales
    for yscale in yscales_diff:
        ax.set_yscale(yscale)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{path_outputs_model}/spectrumDiff_revFinal-{revfinal}_yscale-{yscale}.png", dpi=200)
            
    plt.close()

# Wave number error analysis
for revfinal in [len(its_timesteps)]:
    # Calculate FFT of true data
    true_data_fft = rfft_abs_mirror_torch(train_truth_use, axis=3).mean(axis=0).to(device)
    true_data_fft_mean = torch.mean(true_data_fft, dim=1).to(device)

    ks = np.arange(true_data_fft_mean.shape[1])
    startk = 2
    ks_keep = ks[startk:]

    # Initialize array to store error for each channel, k, and timestep
    error_by_k_timestep = np.zeros((len(chs), len(ks), len(its_timesteps)))

    # Calculate error for each timestep
    for ich in range(len(chs)):  ### loop over channels ###
        for it, t in its_timesteps:
            probe_data_fft = rfft_abs_mirror_torch(probe_data[:, it, 0], axis=3).mean(axis=0)
            probe_data_fft_mean = torch.mean(probe_data_fft, dim=1).to(device)
            error_by_k_timestep[0, :, it] = c2n((probe_data_fft_mean[0] - true_data_fft_mean[0, :]).abs())
        



 # Plot per-channel wave number error vs timesteps
    for ich, ch in enumerate(chs):  ### plot for each channel separately ###
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for k in ks_keep:
            ax.plot(its, error_by_k_timestep[ich, k, its],
                    color=cm.rainbow_r(k / ks[-1]), alpha=0.5)
        # Add colorbar for k values
        cbar = fig.colorbar(cm.ScalarMappable(cmap='rainbow_r'), ax=ax)
        numticks = 11
        kticks_use = np.linspace(ks[-1], ks[0], num=numticks).astype(int)
        kticks_use_norm = (kticks_use)/ks[-1]
        cbar.set_ticks(kticks_use_norm)
        cbar.set_ticklabels([f"{k}" for k in kticks_use])
        cbar.set_label('Wave number k')

        ax.set_title(f"{ch}")
        ax.set_xlabel('1 - tau (inverse diffusion step)')
        ax.set_ylabel('Amplitude Error')
        ax.grid(alpha=0.3)

        plt.suptitle(f"Probe data FFT, difference from true vs timesteps, for each k: {ch}")

        # Save with different y-scales
        for yscale in yscales_diff:
            ax.set_yscale(yscale)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{path_outputs_model}/kDiff_{ch}_revFinal-{revfinal}_yscale-{yscale}.png", dpi=200)
                
        plt.close()

# Noise spectrum analysis
for revfinal in [len(its_timesteps)]:
    startk = 2

    # Calculate FFT of true data for reference
    true_data_fft = rfft_abs_mirror_torch(train_truth_use, axis=3).mean(axis=0)
    true_data_fft_mean = torch.mean(true_data_fft, dim=1)
    ks = np.arange(true_data_fft_mean.shape[1])
    
    for ich, ch in enumerate(chs):  ### loop over channels ###
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        # Plot noise spectrum for each timestep
        for it, t in its_timesteps_plot:
            noise_fft = rfft_abs_mirror_torch(probe_noise[:, it, 0], axis=3).mean(axis=0)
            noise_fft_mean = torch.mean(noise_fft, dim=1)

        # Add legend entry for selected timesteps
        if (it % (len(its_timesteps)//3)) == 0:
            ax.plot([], [], color=cm.rainbow(it/its_timesteps.shape[0]), label=f"reverse t: {t}")

        # Plot noise spectrum
        ax.plot(ks[startk:], c2n(noise_fft_mean[0, startk:]), 
                color=cm.rainbow(it/its_timesteps.shape[0]), alpha=0.5)
    
    # Plot final timestep noise spectrum
    final_noise_fft = rfft_abs_mirror_torch(probe_noise[:, itrevfinal, 0], axis=3).mean(axis=0)
    final_noise_fft_mean = torch.mean(final_noise_fft, dim=1)


    # Add colorbar for timesteps
    cbar = fig.colorbar(cm.ScalarMappable(cmap='rainbow'), ax=ax)
    numticks = 11
    cbar.set_ticks(np.linspace(0, 1, num=numticks))
    cbar.set_ticklabels([f"{int(t)}" for t in np.linspace(ftausteps, stausteps, num=numticks)])
    cbar.set_label('Reverse timestep')

    ax.set_title(f"{ch}")
    ax.legend()
    ax.set_xlabel('k')
    ax.set_ylabel('Amplitude')
    ax.grid(alpha=0.3)
    
    plt.suptitle(f"Noise spectrum\n{ch} | Reverse timesteps from {ftausteps} to {stausteps}")
    plt.tight_layout()

    # Save with different y-scales
    for yscale in yscales:
        ax.set_yscale(yscale)
        plt.tight_layout()
        plt.savefig(f"{path_outputs_model}/spectrumNoise_revFinal-{revfinal}_yscale-{yscale}.png", dpi=200)

    plt.close()


# Noise distribution analysis
for revfinal in [len(its_timesteps)]:
    bins = 500
    pdf_range = (-4, 4)

    for ich, ch in enumerate(chs):  ### loop over channels ###
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        # Plot noise distribution for each timestep
        for it, t in its_timesteps_plot2:
            noise_flat = c2n(probe_noise[:, it, 0, ich]).flatten()
            hist, bin_edges = np.histogram(noise_flat, bins=bins, range=pdf_range, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]
            hist = hist / (hist.sum() * bin_width)  # normalize
            ax.plot(bin_centers, hist, color=cm.rainbow(it / its_timesteps.shape[0]), alpha=0.5)

        # Plot standard normal PDF for comparison
        x = np.linspace(pdf_range[0], pdf_range[1], bins)
        pdf = norm.pdf(x, loc=0, scale=1)
        ax.plot(x, pdf, color="black", alpha=1.0, label="Gaussian")

        # Add colorbar
        cbar = fig.colorbar(cm.ScalarMappable(cmap='rainbow'), ax=ax)
        numticks = 11
        cbar.set_ticks(np.linspace(0, 1, num=numticks))
        cbar.set_ticklabels([f"{int(t)}" for t in np.linspace(ftausteps, stausteps, num=numticks)])
        cbar.set_label('Reverse timestep')

        ax.set_title(f"{ch}")
        ax.set_xlim(pdf_range)
        ax.legend()
        ax.set_xlabel('Noise')
        ax.set_ylabel('PDF')
        ax.grid(alpha=0.3)

        plt.suptitle(f"Noise distribution | {ch}\nReverse timesteps from {ftausteps} to {stausteps}")

        # Save with different y-scales
        for yscale in ["linear", "log"]:
            ax.set_yscale(yscale)
            plt.tight_layout()
            plt.savefig(f"{path_outputs_model}/histNoise_{ch}_revFinal-{revfinal}_yscale-{yscale}.png", dpi=200)

        plt.close()



