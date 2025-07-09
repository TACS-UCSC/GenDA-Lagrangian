import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
from loss_Spectrum import spectral_sqr_abs2

LossFunction= "MSE"  # Loss funtion is either MSE or Spectrum
EPOCH=500
DATA="dn" #Denorm and norm task, the normalization part changed.
MODELNAME=DATA+LossFunction+'UNET90_EP'+str(EPOCH)
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
print("Startsparsity",u_train_m.shape, u_train_o.shape, u_test_m.shape, u_test_o.shape)

# Split training data into training and validation sets
total_train_samples = u_train_m.shape[0]
val_size = int(0.2 * total_train_samples)  # 20% for validation
train_size = total_train_samples - val_size

train_dataset, val_dataset = random_split(TensorDataset(u_train_m, u_train_o), [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


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

if LossFunction == "MSE":
    criterion = nn.MSELoss()
elif LossFunction== "Spectrum_MSE":
    criterion= spectral_sqr_abs2

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping criteria
patience = 20
best_val_loss = float('inf')
epochs_no_improve = 0
losslist = []
val_loss_list = []

# Training loop with early stopping
for epoch in range(EPOCH):  # High epoch count as formal; stops early if necessary
    model.train()
    epoch_losses = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    losslist.append(avg_epoch_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
    avg_val_loss = val_loss / len(val_loader)
    val_loss_list.append(avg_val_loss)

    print(f'Epoch: {epoch + 1}, Training Loss: {avg_epoch_loss:.6f}, Validation Loss: {avg_val_loss:.6f}')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODELNAME+'.pth')  # Save best model
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

