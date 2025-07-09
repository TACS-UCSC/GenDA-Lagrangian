import torch
import torch.nn as nn


torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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



