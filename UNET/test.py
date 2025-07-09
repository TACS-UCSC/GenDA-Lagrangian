import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import math


torch.manual_seed(0)
np.random.seed(0)
print(torch.__version__)

LossFunction= "MSE"  # Loss funtion is either MSE or Spectrum
EPOCH=500
DATA="dn" #Denorm and norm task, the normalization part changed.
MODELNAME=DATA+LossFunction+'Unet99S_EP'+str(EPOCH)
print("MODELNAME:"+MODELNAME)
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Load data
from load_data_CNN import get_dynamics_data
u_train_m, u_train_o, u_test_m, u_test_o = get_dynamics_data()

u_train_m = u_train_m.to(device)
u_train_o = u_train_o.to(device)
u_test_m = u_test_m.to(device)
u_test_o = u_test_o.to(device)
print("Testcheck 99 Sparsity",u_train_m.shape, u_train_o.shape, u_test_m.shape, u_test_o.shape)




def min_max_denormalize(normalized_data, min_val, max_val):
    """
    Denormalize data scaled with Min-Max normalization to its original scale.

    Parameters:
        normalized_data (array): The normalized data.
        min_val (float): The minimum value of the original data before normalization.
        max_val (float): The maximum value of the original data before normalization.

    Returns:
        original_data (array): The denormalized data.
    """
    data = normalized_data * (max_val - min_val) + min_val
    return data

min_val = -47.3168934  
max_val =  48.1056251


# Define the Unet model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # ---- Encoder ----
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.down1 = conv_block(2, 64)         # Encoder block 1
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = conv_block(64, 128)       # Encoder block 2
        self.pool2 = nn.MaxPool2d(2)

        # ---- Bottleneck ----
        self.bottleneck = conv_block(128, 256) # Bottleneck block

        # ---- Decoder ----
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = conv_block(256, 128)    # Decoder block 1 (up from bottleneck + skip from down2)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = conv_block(128, 64)     # Decoder block 2 (up from above + skip from down1)

        # ---- Output ----
        self.final = nn.Conv2d(64, 1, kernel_size=1)  # Final conv to produce 1 output channel

    def forward(self, x):
        # ---- Encoder forward ----
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        # ---- Bottleneck forward ----
        bn = self.bottleneck(p2)

        # ---- Decoder forward ----
        u2 = self.up2(bn)
        cat2 = torch.cat([u2, d2], dim=1)
        u2 = self.upconv2(cat2)

        u1 = self.up1(u2)
        cat1 = torch.cat([u1, d1], dim=1)
        u1 = self.upconv1(cat1)

        return self.final(u1)


model = UNet().to(device)


criterion = nn.MSELoss()


# Path where the model is saved
# model_path = "/glade/derecho/scratch/nasefi/compressed_sensing/dnMSEUnet99S_EP500.pth"
model_path = "/glade/derecho/scratch/nasefi/compressed_sensing/Beta_channel-turbulence_n/save_models/models_turb2d/denorm_new/dnMSEUnet99S_EP500.pth"

# Load the saved model's state dict
model.load_state_dict(torch.load(model_path))
model.eval()


# Test the model
test_dataset = TensorDataset(u_test_m, u_test_o)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
total_loss = 0

total_loss_denorm = 0
all_outputs_denorm = []
all_labels_denorm = []

# Loop over all test data
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        outputs_denorm = min_max_denormalize(outputs.cpu(), -47.3168934, 48.1056251)
        labels_denorm = min_max_denormalize(labels.cpu(), -47.3168934, 48.1056251)

        lossdenorm= criterion(outputs_denorm, labels_denorm)
        total_loss_denorm += lossdenorm.item()

        all_outputs_denorm.append(outputs_denorm)
        all_labels_denorm.append(labels_denorm)


# average_loss = total_loss / len(test_loader)
# sqrt_of_average_loss = math.sqrt(average_loss)


# average_loss_denorm = total_loss_denorm/ len(test_loader)
# sqrt_of_average_loss_denorm = math.sqrt(average_loss_denorm)


# print(f"Average MSE Test Loss: {average_loss:.6f}")
# print(f"Average RMSE Test Loss: {sqrt_of_average_loss:.6f}")

# print(f"Average MSE Test Loss_denorm: {average_loss_denorm:.6f}")
# print(f"Average RMSE Test Loss_denorm: {sqrt_of_average_loss_denorm:.6f}")

# Concatenate all batch outputs and labels
all_outputs_denorm = torch.cat(all_outputs_denorm, dim=0)
all_labels_denorm = torch.cat(all_labels_denorm, dim=0)

outputs_100 = all_outputs_denorm[:100]
labels_100 = all_labels_denorm[:100]

rmse_100 = torch.sqrt(torch.mean((outputs_100 - labels_100) ** 2))
print(f" RMSE for first 100 samples (denormalized): {rmse_100.item():.6f}")




# def plot_fixed_index(data_loader, model, device, index, min_val, max_val):
#     model.eval()
#     with torch.no_grad():
#         for images, labels in data_loader:
#             if images.size(0) > index:  # Ensure the batch is large enough
#                 image = images[index].unsqueeze(0).to(device)
#                 label = labels[index].unsqueeze(0).to(device)
#                 output = model(image)

#                 # image_denorm = min_max_denormalize(image.cpu(),  min_val, max_val)
#                 output_denorm = min_max_denormalize(output.cpu(),  min_val, max_val)
#                 label_denorm = min_max_denormalize(label.cpu(),  min_val, max_val)

#                 imageT = image.transpose(2,3)
#                 outputT = output_denorm.transpose(2,3)
#                 labelT = label_denorm .transpose(2,3)

#                 input_vis = imageT.cpu().squeeze()[0]  # Visualize the first channel #image has 2 channel, label and output has 1

#                 fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
#                 cmap = plt.get_cmap('coolwarm')

#                 input_plot = axes[0].imshow(input_vis, cmap=cmap )
#                 axes[0].set_title('Input')
#                 # fig.colorbar(input_plot, ax=axes[0], fraction=0.046, pad=0.04)

#                  # Overlay markers on non-zero values
#                 non_zero_coords = np.where(input_vis != 0)  # Get indices of non-zero values
#                 axes[0].scatter(non_zero_coords[1], non_zero_coords[0], color='red', s=1)  # Mark non-zero points

#                 output_plot = axes[1].imshow(outputT.cpu().squeeze(), cmap=cmap )
#                 axes[1].set_title('Predicted Output')
#                 fig.colorbar(output_plot, ax=axes[1], fraction=0.046, pad=0.04)

#                 label_plot = axes[2].imshow(labelT.cpu().squeeze(), cmap=cmap )
#                 axes[2].set_title('Actual Output')
#                 fig.colorbar(label_plot, ax=axes[2], fraction=0.046, pad=0.04)

#                 plt.savefig(MODELNAME+str(index)+'.png')  # Dynamic filename based on index
#                 plt.show()
#                 plt.close(fig)  # Close the figure to free up memory
#                 break  # Stop after the first batch
#             else:
#                 print(f"Batch size is smaller than the specified index {index}.")
#                 break

# plot_fixed_index(test_loader, model, device, index=1,  min_val= -47.3168934, max_val = 48.1056251)  # Plot the first index of a batch
# plot_fixed_index(test_loader, model, device, index=5, min_val= -47.3168934, max_val = 48.1056251) 
# plot_fixed_index(test_loader, model, device, index=10, min_val= -47.3168934, max_val = 48.1056251)  
# plot_fixed_index(test_loader, model, device, index=15, min_val= -47.3168934, max_val = 48.1056251)  
# plot_fixed_index(test_loader, model, device, index=20, min_val= -47.3168934, max_val = 48.1056251) 


# def compute_spectrum_torch(data):
#     data_torch = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data.float()
#     return torch.abs(torch.fft.rfft(data_torch, dim=-1)).numpy()

# def compute_mean_std_spectrum_torch(data):
#     data_torch = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data.float()
#     magnitude_spectrum = torch.abs(torch.fft.rfft(data_torch, dim=2))
#     return magnitude_spectrum.mean(1).mean(0).numpy(), magnitude_spectrum.mean(1).std(0).numpy()

# #  [batch, latitude, long]

# def spectrum_rmse(pred_data, target_data):
#     pred_spectrum, _ = compute_mean_std_spectrum_torch(pred_data)
#     target_spectrum, _ = compute_mean_std_spectrum_torch(target_data)
#     return np.sqrt(np.mean((pred_spectrum - target_spectrum) ** 2))

# def compute_spectrum_niloo(data):
#     data_torch = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data.float()
#     magnitude_spectrum = torch.abs(torch.fft.rfft(data_torch, dim=1))
#     return magnitude_spectrum.mean(2).mean(0).numpy(), magnitude_spectrum.mean(0).std(0).numpy()


# def compute_spectrum_c(data):
#     data_torch = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data.float()
#     magnitude_spectrum = torch.abs(torch.fft.rfft(data_torch, dim=1))
#     return magnitude_spectrum.mean(0).numpy(), magnitude_spectrum.mean(0).std(0).numpy()

# #  [batch, latitude, long]


# # Assuming `outputs` and `originals` are numpy arrays with shape [3000, 1, 256, 256]
# # I removed the channed dimension here. 

# #one time step. 

# print("Aggregated Outputs Shape:", all_outputs_denorm.shape)
# print("Aggregated Originals Shape:", all_labels_denorm.shape)
# # Assuming you have functions to compute spectrum defined
# outputs_mean_spectrum, _ = compute_mean_std_spectrum_torch(all_outputs_denorm[:,0].float().cpu())
# originals_mean_spectrum, _ = compute_mean_std_spectrum_torch(all_labels_denorm[:,0].float().cpu())

# # Initialize lists to store outputs and labels for all batches


# # Plot the spectrum FFT

# plt.figure(figsize=(10, 5))
# plt.loglog(outputs_mean_spectrum, color='blue', label='Mean Output Spectrum')
# plt.loglog(originals_mean_spectrum, color='black', label='Mean Original Spectrum')
# plt.title('Power Spectrum for '+MODELNAME)
# plt.xlabel('Wavenumber')
# plt.ylabel('Magnitude')
# plt.legend()
# plt.grid(True)
# plt.savefig(MODELNAME+"FFT.png")
# plt.show()

# # outputs_mean_spectrum, _ = compute_spectrum_niloo(all_outputs[:,0].float().cpu())
# # originals_mean_spectrum, _ = compute_spectrum_niloo(all_labels[:,0].float().cpu())

# plt.figure(figsize=(10, 5))
# plt.loglog(outputs_mean_spectrum, color='blue', label='Mean Output Spectrum')
# plt.loglog(originals_mean_spectrum, color='black', label='Mean Original Spectrum')
# plt.title('Power Spectrum for '+MODELNAME)
# plt.xlabel('Wavenumber')
# plt.ylabel('Magnitude')
# plt.legend()
# plt.grid(True)
# plt.savefig(MODELNAME+"niloo"+"FFT.png")
# plt.show()

# #3000, 256, 256 

# outputs_mean_spectrum, _ = compute_spectrum_c(all_outputs_denorm[20,0].float().cpu())
# originals_mean_spectrum, _ = compute_spectrum_c(all_labels_denorm[20,0].float().cpu())

# plt.figure(figsize=(10, 5))
# plt.loglog(outputs_mean_spectrum, color='blue', label='Mean Output Spectrum')
# plt.loglog(originals_mean_spectrum, color='black', label='Mean Original Spectrum')
# plt.title('Power Spectrum for '+MODELNAME)
# plt.xlabel('Wavenumber')
# plt.ylabel('Magnitude')
# plt.legend()
# plt.grid(True)
# plt.savefig(MODELNAME+"new20"+"FFT.png")
# plt.show()

