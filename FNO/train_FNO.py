################################################################
# training FNO model
################################################################
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
from Fno2D import *
from data_loader_Fno import get_dynamics_data
from utilities3 import *
from count_trainable_params import count_parameters
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
from Loss_Spectrum import spectral_sqr_abs2

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


# Split training data into training and validation sets
total_train_samples = u_train_m.shape[0]
val_size = int(0.2 * total_train_samples)  # 20% for validation
train_size = total_train_samples - val_size

train_dataset, val_dataset = random_split(TensorDataset(u_train_m, u_train_o), [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Configuration

learning_rate = 0.001
modes = 129  #since  Number of Fourier modes to multiply, at most floor(N/2) + 1. 
width = 20  

# Initialize the FNO model
net = FNO2d(modes, modes, width).to(device)
total_params = count_parameters(net)
print("total_params for FNO2d", total_params)


 # Loss funtion is either MSE or Spectrum

if LossFunction == "MSE":
    criterion = nn.MSELoss()
elif LossFunction== "Spectrum_MSE":
    criterion= spectral_sqr_abs2

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)



# Early stopping criteria
patience = 20
best_val_loss = float('inf')
epochs_no_improve = 0
losslist = []
val_loss_list = []

# Training loop with early stopping
for epoch in range(EPOCH):  
    net.train()
    epoch_losses = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    
    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    losslist.append(avg_epoch_loss)

    net.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            val_loss += criterion(outputs, labels).item()
    avg_val_loss = val_loss / len(val_loader)
    val_loss_list.append(avg_val_loss)

    print(f'Epoch: {epoch + 1}, Training Loss: {avg_epoch_loss:.6f}, Validation Loss: {avg_val_loss:.6f}')
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(net.state_dict(), MODELNAME+'.pth')  
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print('Early stopping!')
            break

print("Finished Training or Stopped Early due to Non-Improvement")

loss_data = pd.DataFrame({
    'Epoch': range(1, len(losslist) + 1),
    'Training Loss': losslist,
    'Validation Loss': val_loss_list
})
loss_data.to_csv(MODELNAME+'.csv', index=False)


# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(losslist, label='Training Loss', marker='o')
plt.plot(val_loss_list, label='Validation Loss', marker='x')
plt.title('Training and Validation Loss of '+ MODELNAME)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(MODELNAME+'.png')
plt.show()
