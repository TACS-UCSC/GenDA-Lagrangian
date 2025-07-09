from importlib import reload
import logging
import sys
import yaml
import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from pprint import pformat

# Load Setup

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
# from models import Unet2d                
# reload(Unet2d)
reload(simple_unet)
reload(fno2d)
from models.simple_unet import SimpleUnet, SimpleUnetCond
from models.fno2d import FNO2D_grid, FNO2D_grid_tembedding, FNO2D_grid_tembedding_cond
from models.Unet2d import UNet         
from models import loss_functions
reload(loss_functions)
from models.loss_functions import LOSS_FUNCTIONS
import utilities
reload(utilities)
from utilities import n2c, c2n, pthstr, linear_beta_scheduler, cosine_beta_scheduler
import metrics
reload(metrics)


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

NAME="FNO_DDPM_99S_Ep500"
# Generate model name if not provided
if model_name is None:
    model_name = f"{NAME}-ddpm_arch-{ddpm_arch}_time-{current_time}_timesteps-{timesteps}_epochs-{epochs}"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler(f"{logging_dir}/ddpm_qgm_losses_{current_time}.log"),
    logging.StreamHandler()
])
printlog = logging.info
printlog("-"*40)
printlog(f"Running ddpm_turb2d.py for {model_name}...")
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


# # Load training data

train_sparse_loc = "/glade/derecho/scratch/nasefi/compressed_sensing/Beta_channel-turbulence_n/save_models/dnMSEFNO99S_EP500/train_input_sparse.pkl"
train_pred_loc = "/glade/derecho/scratch/nasefi/compressed_sensing/Beta_channel-turbulence_n/save_models/dnMSEFNO99S_EP500/train_pred_sparse.pkl"
truth_loc = "/glade/derecho/scratch/nasefi/compressed_sensing/Beta_channel-turbulence_n/save_models/dnMSEFNO99S_EP500/train_truth.pkl"


# Load data files
with open(train_sparse_loc, "rb") as f:
    train_input_sparse = pickle.load(f)

with open(train_pred_loc, "rb") as f:
    train_pred_sparse = pickle.load(f)

with open(truth_loc, "rb") as f:
    truth_train = pickle.load(f)

print(train_input_sparse.shape, train_pred_sparse.shape, truth_train.shape)


# # FOR FNO Channle Permute dimensions to [batch, channel, height, width]
train_input_sparse = train_input_sparse.permute((0, 3, 1, 2))
train_pred_sparse = train_pred_sparse.permute((0, 3, 1, 2))
truth_train = truth_train.permute((0, 3, 1, 2))


# Create grid coordinates
xs = torch.linspace(0, 1, Nx)
ys = torch.linspace(0, 1, Ny)

# Reload modules to ensure latest versions
reload(models)
reload(simple_unet)


# Initialize model based on architecture
if ddpm_arch == "unet":
    model = SimpleUnet(**ddpm_params).to(device)
elif ddpm_arch == "unet_cond":
    model = SimpleUnetCond(**ddpm_params).to(device)
elif ddpm_arch == "fno2d":
    model = FNO2D_grid_tembedding_cond(**ddpm_params).to(device)
    model.gridx = xs.to(device)
    model.gridy = ys.to(device)
    
# Initialize optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr)

# Test model with a small batch
t = torch.randint(0, timesteps, (batch_size,), device=device).long()
data = truth_train
cond_data = train_pred_sparse
_ = model(data[:batch_size], cond_data[:batch_size], t)

hyperparam_dict["epochs_run"] = 0

##############
## TRAINING ##
##############

if True:
    printlog(f"Training {model_name}...")
    
    # Initialize noise scheduler
    if beta_scheduler == "linear":
        betas, alphas, alphas_cumprod = linear_beta_scheduler(beta_start, beta_end, timesteps, device=device)
    elif beta_scheduler == "cosine":
        betas, alphas, alphas_cumprod = cosine_beta_scheduler(timesteps, device=device)

    # Plot alphas_cumprod for visualization
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(timesteps), c2n(alphas_cumprod), label='alphas_cumprod')
    plt.xlabel('Timesteps')
    plt.ylabel('Alphas Cumulative Product')
    plt.title(f'Alphas Cumulative Product over Timesteps\nbeta_start: {beta_start}, beta_end: {beta_end}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{model_dir}/alphas_cumprod.png", dpi=200)
    plt.close()

    # Initialize tracking variables
    if "epochs_run" not in hyperparam_dict:
        hyperparam_dict["epochs_run"] = 0

    loss_batch = []
    loss_epoch = []
    ibatch = 0
    
   
    best_loss = float("inf")
    epochs_since_improvement = 0
    patience = 10  # Stop if no improvement after 10 epochs


    # Training loop
    for epoch in range(epochs):
        printbatch = 0
        for batch_num in range(0, data.shape[0], batch_size):
            # Get batch data
            data_batch = data[batch_num:batch_num+batch_size].to(device)
            cond_batch = cond_data[batch_num:batch_num+batch_size].to(device)
            batch_size_actual = data_batch.shape[0]

            if train_type == "noise":
                # Sample random timesteps
                t = torch.randint(0, timesteps, (batch_size_actual,), device=device)

                # Add noise to data
                noise = torch.randn_like(data_batch)
                noisy_data = torch.sqrt(alphas_cumprod[t].view(-1, 1, 1, 1)) * data_batch + \
                             torch.sqrt(1 - alphas_cumprod[t].view(-1, 1, 1, 1)) * noise

                # Train model to predict noise
                optimizer.zero_grad()
                predicted_noise = model(noisy_data, cond_batch, t)
                
                # Select loss function based on batch number
                if ibatch <= loss_function_start_batch or loss_function_start_batch == -1:
                    loss_use = loss_function_start
                    loss = LOSS_FUNCTIONS[loss_use](predicted_noise.permute((0, 2, 3, 1)), 
                                                   noise.permute((0, 2, 3, 1)), 
                                                   **loss_args_start)
                else:
                    loss_use = loss_function
                    loss = LOSS_FUNCTIONS[loss_function](predicted_noise.permute((0, 2, 3, 1)), 
                                                        noise.permute((0, 2, 3, 1)), 
                                                        **loss_args_end)

                loss.backward()
                optimizer.step()

            elif train_type == "tauFull":
                # Full timestep sampling from timesteps to 0
                tau0 = torch.zeros(batch_size_actual, device=device).long()
                tau1 = torch.full((batch_size_actual,), timesteps-1, device=device).long()
                rev_steps = timesteps

                # Add noise for tau0 and tau1 timesteps
                noise = torch.randn_like(data_batch)
                data_noise_rev = torch.sqrt(alphas_cumprod[tau0].view(-1, 1, 1, 1)) * data_batch + \
                                 torch.sqrt(1 - alphas_cumprod[tau0].view(-1, 1, 1, 1)) * noise
                data_noise = torch.sqrt(alphas_cumprod[tau1].view(-1, 1, 1, 1)) * data_batch + \
                             torch.sqrt(1 - alphas_cumprod[tau1].view(-1, 1, 1, 1)) * noise

                # Train model through reverse process
                optimizer.zero_grad()
                x = data_noise

                for tr in range(rev_steps):
                    tau = tau1 - tr  # Reverse from last timestep
                    predicted_noise = model(x, tau)
                    alpha_t = alphas[tau].view(-1, 1, 1, 1)
                    alpha_bar_t = alphas_cumprod[tau].view(-1, 1, 1, 1)
                    x = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise)
                    
                    # Add noise only if tau > 0
                    teff = torch.max(torch.ones(tau.shape, device=device), tau).long()
                    beta_t = betas[teff].view(-1, 1, 1, 1)
                    step_add_noise = (tau.view(-1, 1, 1, 1) > 0).int()
                    x = x + torch.sqrt(beta_t) * torch.randn_like(x, requires_grad=True) * step_add_noise

                # Select loss function based on batch number
                if ibatch < loss_function_start_batch:
                    loss_use = loss_function_start
                    loss = LOSS_FUNCTIONS[loss_use](x.permute((0, 2, 3, 1)), 
                                                   data_noise_rev.permute((0, 2, 3, 1)), 
                                                   **loss_args_start)
                else:
                    loss_use = loss_function
                    loss = LOSS_FUNCTIONS[loss_use](x.permute((0, 2, 3, 1)), 
                                                   data_noise_rev.permute((0, 2, 3, 1)), 
                                                   **loss_args_end)

                loss.backward()
                optimizer.step()

            # Track losses
            loss_batch.append([ibatch, loss.item()])

            # Print progress
            if ibatch >= printbatch:
                printlog(f"Epoch [{epoch+1}/{epochs}], ibatch {ibatch+1}, loss_use: {loss_use}, Loss: {loss.item():.8f}")
                printbatch = ibatch + 10
            
            ibatch += 1

        # Track epoch loss
        loss_epoch.append([ibatch, loss.item()])
        current_epoch_loss = loss.item()

        if current_epoch_loss < best_loss:
            best_loss = current_epoch_loss
            torch.save(model.state_dict(), f"{model_dir}/{model_name}_best.pth")  # Save best model here
            epochs_since_improvement = 0
            printlog(f"✨ New best loss: {best_loss:.8f}")
        else:
            epochs_since_improvement += 1
            printlog(f"No improvement for {epochs_since_improvement} epoch(s).")

        # Early stopping check
        if epochs_since_improvement >= patience:
            printlog(f" Early stopping triggered after {epoch+1} epochs.")
            break

        # Plot training progress
        loss_batch_arr = np.array(loss_batch)
        loss_epoch_arr = np.array(loss_epoch)
        plt.plot(loss_batch_arr[:,0], loss_batch_arr[:,1], color="blue", label="batch loss", alpha=0.5)
        plt.scatter(loss_epoch_arr[:,0], loss_epoch_arr[:,1], color="red", label="epoch loss")
        plt.xlabel("Batch number")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid(alpha=0.3)
        plt.legend(loc="upper right")
        plt.savefig(f"{model_dir}/loss_batch_epoch.png", dpi=200)
        plt.close()
        
        
        # Initialize or load checkpoint dictionary
        checkpoint_path = f"{model_dir}/{model_name}_all_epochs.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = {"epochs_run": 0, "models": {}}

        # Save model for the current epoch in the dictionary
        checkpoint["models"][f"epoch-{epoch+1}"] = model.state_dict()
        checkpoint["epochs_run"] = epoch + 1

        # Save the entire checkpoint dictionary
        torch.save(checkpoint, checkpoint_path)

        # (Optional) Still save the last epoch separately if needed
        torch.save(model.state_dict(), f"{model_dir}/{model_name}_last_epoch.pth")

        # Update the config file
        hyperparam_dict["epochs_run"] += 1
        with open(f"{model_dir}/config.yml", 'w') as h:
            yaml.dump(hyperparam_dict, h, default_flow_style=False)


else:
    # Code for loading a pre-trained model (not used in current run)
    model_name = "ddpm_arch-unet_time-2025-02-21-01-25_timesteps-1000_epochs-80_epoch-79"
    model_loc = "/glade/derecho/scratch/llupinji/diffusion_qgm_outputs/models_ddpm_specLoss/ddpm_arch-unet_time-2025-02-21-01-25_timesteps-1000_epochs-80/ddpm_arch-unet_time-2025-02-21-01-25_timesteps-1000_epochs-80_epoch-79.pt"
    model.load_state_dict(torch.load(model_loc))

    # Load config for pre-trained model
    config_path = "/glade/derecho/scratch/llupinji/diffusion_qgm_outputs/models_ddpm_specLoss/ddpm_arch-unet_time-2025-02-21-01-25_timesteps-1000_epochs-80/config.yml"
    with open(config_path, 'r') as h:
        hyperparam_dict = yaml.load(h, Loader=yaml.FullLoader)

    # Extract parameters from loaded config
    timesteps = hyperparam_dict["timesteps"]
    beta_start = hyperparam_dict["beta_start"]
    beta_end = hyperparam_dict["beta_end"]
    batch_size = hyperparam_dict["batch_size"]
    epochs = hyperparam_dict["epochs"]
    beta_scheduler = hyperparam_dict["beta_scheduler"]
    epochs_run = hyperparam_dict["epochs_run"]

    # Initialize noise scheduler
    if beta_scheduler == "linear":
        betas, alphas, alphas_cumprod = linear_beta_scheduler(beta_start, beta_end, timesteps, device=device)
    elif beta_scheduler == "cosine":
        betas, alphas, alphas_cumprod = cosine_beta_scheduler(timesteps, device=device)

