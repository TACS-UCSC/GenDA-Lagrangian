import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding='same')
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1) #maybe change the dimension, check.
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class BlockCond(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding='same')
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
            self.conv1_cond = nn.Conv2d(in_ch, out_ch, 3, padding='same')
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            self.conv1_cond = nn.Conv2d(in_ch, out_ch, 3, padding='same')

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.bnorm1_cond = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        self.relu_cond = nn.ReLU()
        
    def forward(self, x, cond, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        hcond = self.bnorm1_cond(self.relu_cond(self.conv1_cond(cond)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel

        h = h + time_emb
        h = h + hcond

        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, 
                 in_channels = 4, 
                 out_channels = 4, 
                 down_channels = (64, 64, 64, 64, 64), 
                 up_channels = (64, 64, 64, 64, 64), 
                 time_emb_dim = 64):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # down_channels = (64, 64, 64, 64, 64)
        # up_channels = (64, 64, 64, 64, 64)
        # down_channels = (128, 128, 128, 128, 128)
        # up_channels = (128, 128, 128, 128, 128)
        # # time_emb_dim = 32
        # time_emb_dim = 64
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding='same')

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])


        self.output = nn.Conv2d(up_channels[-1], out_channels, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)


    
class SimpleUnetCond(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, 
                 in_channels = 4, 
                 out_channels = 4, 
                 down_channels = (128, 128, 128, 128, 128), 
                 up_channels = (128, 128, 128, 128, 128), 
                 time_emb_dim = 64):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding='same')
        self.conv0_cond = nn.Conv2d(in_channels, down_channels[0], 3, padding='same')

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        for i in range(len(down_channels)-1):
            print("debug ", down_channels[i], " ", down_channels[i+1], " ", time_emb_dim)

        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])


        self.output = nn.Conv2d(up_channels[-1], out_channels, 1)

    def forward(self, x, cond, timestep):
        """
        x is the input to the model
        cond is the condition to the model
        timestep is the timestep to the model
        """
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # condition conv
        cond = self.conv0_cond(cond)

        # add condition to x
        x = x + cond

        #add MLP
        num_blocks = len(self.down_channels)-1
        xpad =  ((x.shape[2]//(2**num_blocks))+1)*(2**num_blocks)-x.shape[2]
        ypad = ((x.shape[3]//(2**num_blocks))+1)*(2**num_blocks)-x.shape[3]
        x = F.pad(x, (0,ypad,0,xpad))
        
        # Unet
        residual_inputs = []
        
        for down in self.downs:
            
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        x = x[:,:,:-xpad:, :-ypad]
        outmatrix= self.output(x)
        return outmatrix
    
    