import torch
# import jax.numpy as jnp
import gc

def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()

def numpy_to_cuda(arr):
    return torch.from_numpy(arr).float().cuda()

def cuda_to_numpy(arr):
    return arr.cpu().detach().numpy()

def count_parameters(model):
    """Counts the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cuda_memory_info():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f"reserved: {r/10e6}")
    print(f"allocated: {a/10e6}")
    print(f"free: {f/10e6}")

n2c = numpy_to_cuda
c2n = cuda_to_numpy

def pthstr(s):
    if type(s) is int:
        return str(s).replace("-", "n").replace(".", "p")
    else:
        return str(s)

def linear_beta_scheduler(beta_start, beta_end, timesteps, device="cpu"):
    """
    betas and alphas for the diffusion process, linear noise scheduler
    """
    betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device) # Precompute the cumulative product of all the alpha terms
    return betas, alphas, alphas_cumprod

# def cosine_beta_scheduler(timesteps, device):
#     """
#     cosine beta scheduler, https://www.zainnasir.com/blog/cosine-beta-schedule-for-denoising-diffusion-models/
#     """
#     s = .008
#     ft = lambda t: (((t/timesteps) + s)/(1+s)*torch.pi/2).cos()**2
#     alpha_ts = ft(torch.arange(timesteps+1).float()).to(device)/ft(torch.zeros(timesteps+1)).to(device)
#     betas = torch.min(1 - alpha_ts[1:]/(alpha_ts[:-1]), torch.tile(torch.tensor(.999), alpha_ts[1:].shape).to(device))
#     alphas = 1 - betas
#     alphas_cumprod = torch.cumprod(alphas, dim=0).to(device) # Precompute the cumulative product of all the alpha terms
#     return betas, alphas, alphas_cumprod

def cosine_beta_scheduler(timesteps, s=0.008, device = "cpu"):
    """
    Generates a cosine beta schedule for the given number of timesteps.

    Parameters:
    - timesteps (int): The number of timesteps for the schedule.
    - s (float): A small constant used in the calculation. Default: 0.008.

    Returns:
    - betas (torch.Tensor): The computed beta values for each timestep.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps).to(device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas,0,.9999).to(device)

    alphas = 1 - betas
    return betas, alphas, alphas_cumprod

# def marginal_prob(x, t, beta_min = 0.0001, beta_max = 20, tmax = 1):
#     integral = beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2
#     expint = jnp.exp(-0.5 * integral)
#     mean = expint * x
#     std = jnp.sqrt(1.0 - jnp.exp(-integral))
#     return integral, expint, mean, std