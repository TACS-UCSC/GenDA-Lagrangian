
################################################################
# test FNO
################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import math
from Fno2D import *
from data_loader_Fno import get_dynamics_data
from utilities3 import *
import numpy as np
import torch.nn as nn
torch.manual_seed(0)
np.random.seed(0)
print(torch.__version__)

LossFunction= "MSE" 
EPOCH=200
MODELNAME=LossFunction+'ch_FNO99S_EP'+str(EPOCH)
print("MODELNAME:"+MODELNAME)
torch.set_default_dtype(torch.float32)


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load training and test dataset
u_train_m, u_train_o, u_test_m, u_test_o = get_dynamics_data()
u_train_m, u_train_o = u_train_m.to(device), u_train_o.to(device)
u_test_m, u_test_o = u_test_m.to(device), u_test_o.to(device)

print("load_data", u_train_m.shape, u_train_o.shape, u_test_m.shape, u_test_o.shape)

# Configuration

learning_rate = 0.001
modes = 129  #since  Number of Fourier modes to multiply, at most floor(N/2) + 1. 
width = 20


net = FNO2d(modes, modes, width).to(device)


criterion = nn.MSELoss()

# Path where the model is saved
model_path = "/glade/derecho/scratch/nasefi/compressed_sensing/MSEch_FNO99S_EP200.pth"


# Load the saved model's state dict
net.load_state_dict(torch.load(model_path))
net.eval()


# Test the model
test_dataset = TensorDataset(u_test_m, u_test_o)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
total_loss = 0
all_outputs = []
all_labels = []

# Loop over all test data
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        all_outputs.append(outputs)
        all_labels.append(labels)


average_loss = total_loss / len(test_loader)
sqrt_of_average_loss = math.sqrt(average_loss)
print(f"Average MSE Test Loss: {average_loss:.6f}")
print(f"Average RMSE Test Loss: {sqrt_of_average_loss:.6f}")



def plot_fixed_index(data_loader, model, device, index, channel_dim=3):
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            if images.size(0) > index:  # Ensure the batch is large enough
                image = images[index].unsqueeze(0).to(device)
                label = labels[index].unsqueeze(0).to(device)
                output = model(image)
                if channel_dim == 3:
                    imageT = image.transpose(1,2)
                    labelT = label.transpose(1,2)
                    outputT = output.transpose(1,2)

                elif channel_dim == 1:
                    imageT = image.transpose(2,3)
                    labelT = label.transpose(2,3)
                    outputT = output.transpose(2,3)
                
                image_vis = imageT.select(-1, 0)  # Select the first channel for visualization
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
                axes[0].imshow(image_vis.cpu().squeeze(), cmap='gray')
                axes[0].set_title('Input')
                axes[1].imshow(outputT.cpu().squeeze(), cmap='gray')
                axes[1].set_title('Predicted Output')
                axes[2].imshow(labelT.cpu().squeeze(), cmap='gray')
                axes[2].set_title('Actual Output')
                plt.savefig(MODELNAME+str(index)+'.png')  # Dynamic filename based on index
                plt.show()
                plt.close(fig)  # Close the figure to free up memory
                break  # Stop after the first batch
            else:
                print(f"Batch size is smaller than the specified index {index}.")
                break
# Usage examples:
plot_fixed_index(test_loader, net, device, index=1, channel_dim=3)  # Plot the first index of a batch
plot_fixed_index(test_loader, net, device, index=5, channel_dim=3) 
plot_fixed_index(test_loader, net, device, index=10, channel_dim=3)  
plot_fixed_index(test_loader, net, device, index=15, channel_dim=3)  
plot_fixed_index(test_loader, net, device, index=20, channel_dim=3) 
plot_fixed_index(test_loader, net, device, index=30, channel_dim=3) 






# For FFT Concatenate all batch outputs and labels
all_outputs = torch.cat(all_outputs, dim=0) #(3000, 256, 256, 1)
all_outputs = all_outputs.permute(0, 3, 1, 2)
all_labels = torch.cat(all_labels, dim=0)
all_labels  = all_labels.permute(0, 3, 1, 2)


#Moein Code
def compute_spectrum_torch(data):
    data_torch = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data.float()
    return torch.abs(torch.fft.rfft(data_torch, dim=-1)).numpy()

def compute_mean_std_spectrum_torch(data):
    data_torch = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data.float()
    magnitude_spectrum = torch.abs(torch.fft.rfft(data_torch, dim=-1))
    return magnitude_spectrum.mean(1).mean(0).numpy(), magnitude_spectrum.mean(1).std(0).numpy()

def spectrum_rmse(pred_data, target_data):
    pred_spectrum, _ = compute_mean_std_spectrum_torch(pred_data)
    target_spectrum, _ = compute_mean_std_spectrum_torch(target_data)
    return np.sqrt(np.mean((pred_spectrum - target_spectrum) ** 2))


# Assuming `outputs` and `originals` are numpy arrays with shape [3000, 1, 256, 256]
# I removed the channed dimension here. 


print("Aggregated Outputs Shape:", all_outputs.shape)
print("Aggregated Originals Shape:", all_labels.shape)
# Assuming you have functions to compute spectrum defined
outputs_mean_spectrum, _ = compute_mean_std_spectrum_torch(all_outputs[:,0].float().cpu())
originals_mean_spectrum, _ = compute_mean_std_spectrum_torch(all_labels[:,0].float().cpu())


# Initialize lists to store outputs and labels for all batches


# Plot the spectrum FFT

plt.figure(figsize=(10, 5))
plt.loglog(outputs_mean_spectrum, color='blue', label='Mean Output Spectrum')
plt.loglog(originals_mean_spectrum, color='black', label='Mean Original Spectrum')
plt.title('Power Spectrum for '+MODELNAME)
plt.xlabel('Wavenumber')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.savefig(MODELNAME+"FFT.png")
plt.show()


def compute_spectrum_niloo(data):
    data_torch = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data.float()
    magnitude_spectrum = torch.abs(torch.fft.rfft(data_torch, dim=1))
    return magnitude_spectrum.mean(2).mean(0).numpy(), magnitude_spectrum.mean(0).std(0).numpy()

outputs_mean_spectrum, _ = compute_spectrum_niloo(all_outputs[:,0].float().cpu())
originals_mean_spectrum, _ = compute_spectrum_niloo(all_labels[:,0].float().cpu())

plt.figure(figsize=(10, 5))
plt.loglog(outputs_mean_spectrum, color='blue', label='Mean Output Spectrum')
plt.loglog(originals_mean_spectrum, color='black', label='Mean Original Spectrum')
plt.title('Power Spectrum for '+MODELNAME)
plt.xlabel('Wavenumber')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.savefig(MODELNAME+"dim=1"+"FFT.png")
plt.show()



def compute_spectrum_c(data):
    data_torch = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data.float()
    magnitude_spectrum = torch.abs(torch.fft.rfft(data_torch, dim=1))
    return magnitude_spectrum.mean(0).numpy(), magnitude_spectrum.mean(0).std(0).numpy()


outputs_mean_spectrum, _ = compute_spectrum_c(all_outputs[20,0].float().cpu())
originals_mean_spectrum, _ = compute_spectrum_c(all_labels[20,0].float().cpu())

plt.figure(figsize=(10, 5))
plt.loglog(outputs_mean_spectrum, color='blue', label='Mean Output Spectrum')
plt.loglog(originals_mean_spectrum, color='black', label='Mean Original Spectrum')
plt.title('Power Spectrum for '+MODELNAME)
plt.xlabel('Wavenumber')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.savefig(MODELNAME+"index20"+"FFT.png")
plt.show()

