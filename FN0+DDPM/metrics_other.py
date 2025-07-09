import numpy as np
import pickle
import torch
import numpy as np
import pickle
import torch
import numpy as np

#########################################################################################
##### KE metric for 2d-turb data

def fft2_to_rfft2(a_hat_fft):
    if a_hat_fft.shape[0] % 2 == 0:
        return a_hat_fft[:, :a_hat_fft.shape[1]//2+1]
    else:
        return a_hat_fft[:, :(a_hat_fft.shape[1]-1)//2+1]


def Omega2Psi_spectral(Omega_hat, invKsq):
    lap_Psi_hat = -Omega_hat
    Psi_hat = lap_Psi_hat * (-invKsq)
    return Psi_hat


def Psi2UV_spectral(Psi_hat, Kx, Ky):
    U_hat = (1.j) * Ky * Psi_hat
    V_hat = -(1.j) * Kx * Psi_hat
    return U_hat, V_hat


def Omega2UV_physical(Omega, Kx, Ky, invKsq):
    Nx, Ny = Omega.shape
    Omega_hat = np.fft.rfft2(Omega)
    Psi_hat = Omega2Psi_spectral(Omega_hat, invKsq)
    U_hat, V_hat = Psi2UV_spectral(Psi_hat, Kx, Ky)
    U = np.fft.irfft2(U_hat, s=[Nx, Ny])
    V = np.fft.irfft2(V_hat, s=[Nx, Ny])
    return U, V


def Omega2UV(Omega, Kx, Ky, invKsq):
    return Omega2UV_physical(Omega, Kx, Ky, invKsq)

# ----------------------------
# KE Calculation
# ----------------------------

def omega2ke(Omega, Kx, Ky, invKsq):
    U, V = Omega2UV(Omega, Kx, Ky, invKsq)
    return U**2 + V**2

# ----------------------------
# Grid and wavenumber setup
# ----------------------------

def gridgen(Lx, Ly, Nx, Ny, INDEXING='ij'):
    dx = Lx / Nx
    dy = Ly / Ny
    x = np.linspace(0, Lx - dx, num=Nx)
    y = np.linspace(0, Ly - dy, num=Ny)
    X, Y = np.meshgrid(x, y, indexing=INDEXING)
    return Lx, Ly, X, Y, dx, dy


def initialize_wavenumbers_rfft2(nx, ny, Lx, Ly, INDEXING='ij'):
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx/nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly/ny)
    Kx, Ky = np.meshgrid(kx, ky, indexing=INDEXING)
    Ksq = Kx ** 2 + Ky ** 2
    Ksq[0,0] = 1e16
    invKsq = 1.0 / Ksq
    invKsq[0,0] = 0.0
    Ksq[0,0] = 0.0
    return fft2_to_rfft2(Kx), fft2_to_rfft2(Ky), fft2_to_rfft2(Ksq), fft2_to_rfft2(invKsq)

# ----------------------------
# Compute average KE field
# ----------------------------

def compute_avg_ke_field(data_tensor, N_SAMPLES, Kx, Ky, invKsq, Nx):
    ke_list = []
    for i in range(N_SAMPLES):
        omega = data_tensor[i, 0, :, :].cpu().numpy()
        ke = omega2ke(omega, Kx, Ky, invKsq)
        ke_list.append(ke)
    avg_ke = np.mean(ke_list, axis=0)
    return avg_ke

# ----------------------------
# Example usage
# ----------------------------

# Parameters
Nx = 256
Lx = Ly = 2 * np.pi

# Generate grid
Lx, Ly, X, Y, dx, dy = gridgen(Lx, Ly, Nx, Nx)

# Compute wavenumbers
Kx, Ky, Ksq, invKsq = initialize_wavenumbers_rfft2(Nx, Nx, Lx, Ly)

# Load data
with open("/glade/derecho/scratch/nasefi/diffusion_FNO/Divergence_error/final_turb2d_pkl/test_FNO_pred_sparse.pkl", "rb") as f:
    test_FNO_pred_sparse = pickle.load(f)

test_FNO_pred_sparse_use = test_FNO_pred_sparse.permute(0, 3, 2, 1)

# Compute average KE over 3000 samples
n_fno_ke_avg = compute_avg_ke_field(
    test_FNO_pred_sparse_use,
    N_SAMPLES=3000,
    Kx=Kx,
    Ky=Ky,
    invKsq=invKsq,
    Nx=Nx
)

print("Average KE field shape:", n_fno_ke_avg.shape)



###################################################################################################
#### Strain rate and relative vorticity for Golf of Mexico Dataset

# Define domain and grid for Cartopy
lat_min, lat_max = 17, 31
lon_min, lon_max = -98.99999, -74.08333
ny, nx = 169, 300
lats = np.linspace(lat_min, lat_max, ny)
lons = np.linspace(lon_min, lon_max, nx)
lon2d, lat2d = np.meshgrid(lons, lats)


## Gradient calculations for oceanographic metrics
## Data format: timestep x channel (u, v, ssh) x lat (y coordinate) x lon (x coordinate)

def data_dx(x):
    """
    Calculate x-derivative (longitudinal gradient) of data fields.

    Parameters:
        x (torch.Tensor): Tensor of shape 
                          (B, C, H, W) for 4D 
                          or 
                          (B, T, S, C, H, W) for 6D.

    Returns:
        torch.Tensor: Derivative along the x-axis.
    """
    if x.ndim == 4:
        return torch.gradient(x, dim=3)[0]  # W (x-direction)
    elif x.ndim == 6:
        return torch.gradient(x, dim=5)[0]  # W (x-direction)
    else:
        raise ValueError(f"Unsupported input shape {x.shape}. Expected 4D or 6D tensor.")



def data_dy(x):
    """Calculate y-derivative (latitudinal gradient) of data fields
    Parameters:
        x (torch.Tensor): Tensor of shape 
                          (B, C, H, W) for 4D 
                          or 
                          (B, T, S, C, H, W) for 6D.

    Returns:
        torch.Tensor: Derivative along the y-axis.
    """
    # x is a tensor of shape (batch_size, channel, y, x)
     
    if x.ndim == 4:
        return torch.gradient(x, dim=2)[0] # H (y-direction)
    elif x.ndim ==6:
        return torch.gradient(x, dim=4)[0]     # H (y-direction)
    else:
        raise ValueError(f"Unsopported input shape {x.shape}. Expected 4D or 6D tensor.")




def sigma_n(x):

    """
    Calculate normal component of strain rate tensor.

    Physics: Measures stretching/compression along principal axes.
    Formula: ∂u/∂x − ∂v/∂y

    Parameters:
        x (torch.Tensor): Tensor of shape 
                          (B, C, H, W) or (B, T, S, C, H, W)

    Returns:
        torch.Tensor: Normal strain component.
    """
     
    if x.ndim ==4: 
        return data_dx(x[:,[0],:,:])-data_dy(x[:,[1],:,:])
    elif x.ndim ==6:
        return data_dx(x[:,:, :, [0],:,:])-data_dy(x[:,:, :, [1],:,:])
    else: 
        raise ValueError(f"Unsopported input shape {x.shape}. Expected 4D or 6D tensor.")
 


def sigma_s(x):

    """
    Calculate shear component of strain rate tensor.

    Physics:  Measures deformation due to shearing motion
    Formula: dv/dx + du/dy

    Parameters:
        x (torch.Tensor): Tensor of shape 
                          (B, C, H, W) or (B, T, S, C, H, W)

    Returns:
        torch.Tensor: shear component of strain rate
    """ 
    if x.ndim ==4: 
        return data_dx(x[:,[1],:,:])+data_dy(x[:,[0],:,:])
    elif x.ndim ==6: 
        return data_dx(x[:,:,:,[1],:,:])+data_dy(x[:,:,:,[0],:,:])
    else: 
        raise ValueError(f"Unsopported input shape {x.shape}. Expected 4D or 6D tensor.")
    


def sigma(x):
    """Calculate total strain rate magnitude
    
    Physics: Overall deformation rate of the fluid, combining normal and shear components
    Formula: sqrt(sigma_n^2 + sigma_s^2)
    """
    return torch.sqrt(sigma_n(x)**2+sigma_s(x)**2)



def zeta(x):
    """Calculate relative vorticity
    
    Physics: Measures local rotation of fluid parcels
    Formula: dv/dx - du/dy (curl of velocity field)
    """
    if x.ndim ==4: 
        return data_dx(x[:,[1],:,:])-data_dy(x[:,[0],:,:])
    elif x.ndim ==6: 
        return data_dx(x[:,:, :, [1],:,:])-data_dy(x[:, :, :, [0],:,:])
    else:
        raise ValueError(f"Unsopported input shape {x.shape}. Expected 4D or 6D tensor.")
    


    

