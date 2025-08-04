import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.fft

def get_loss_cond(model, x_0, t, label_batch, loss_func):  #???
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return  loss_func((x_noisy-noise_pred),label_batch ,wavenum_init,lamda_reg)

def mse_loss(output, target):
    loss = torch.mean((output-target)**2)
    return loss

def ocean_loss(output, target, ocean_grid):
    loss = (torch.sum((output-target)**2))/ocean_grid
    return loss

def mseloss_mask(output, target, mask = None, loss_weights = [1,1,1]):
    run_loss = torch.zeros(1).float().cuda()
    loss_weights_norm = np.array(loss_weights)/np.linalg.norm(loss_weights)

    for iw, w in enumerate(loss_weights_norm,0):
        loss_all = (output[...,iw]-target[...,iw])**2
        run_loss += w*torch.mean((output[...,iw]-target[...,iw])**2)

    return run_loss

def spectral_sqr_abs(output, 
                               target, 
                               grid_valid_size = None,
                               wavenum_init_lon = 1, 
                               wavenum_init_lat = 1, 
                               lambda_fft = .5,
                               lat_lon_bal = .5,
                               channels = "all",
                               fft_loss_scale = 1./110.):
        
    """
    Grid and spectral losses, both with mse
    """
    # loss1 = torch.sum((output-target)**2)/ocean_grid + torch.sum((output2-target2)**2)/ocean_grid
    
    ## loss from grid space
    if grid_valid_size is None: 
        grid_valid_size = output[...,[0]].flatten().shape[0]
        
    loss_grid = torch.sum((output-target)**2)/(grid_valid_size*len(channels))
    # loss_grid = torch.mean((output-target)**2)
    # loss1 = torch.abs((output-tnparget))/ocean_grid

    run_loss_run = torch.zeros(1).float().cuda()
    
    if channels == "all":
        num_spectral_chs = output.shape[-1]
        channels = [["_",i,1./num_spectral_chs] for i in np.arange(num_spectral_chs)]
    
    totcw = 0
    for [cname,c,cw] in channels:
        ## it makes sense for me to take fft differences before...if you take mean, you lose more important differences at the equator?
        ## losses from fft, both lat (index 1) and lon (index 2) directions
        ## lat
        out_fft_lat = torch.abs(torch.fft.rfft(output[:,:,:,c],dim=1))[:,wavenum_init_lon:,:]
        target_fft_lat = torch.abs(torch.fft.rfft(target[:,:,:,c],dim=1))[:,wavenum_init_lon:,:]
        loss_fft_lat = torch.mean((out_fft_lat - target_fft_lat)**2)
        ## lon
        out_fft_lon = torch.abs(torch.fft.rfft(output[:,:,:,c],dim=2))[:,:,wavenum_init_lon:]
        target_fft_lon = torch.abs(torch.fft.rfft(target[:,:,:,c],dim=2))[:,:,wavenum_init_lon:]
        loss_fft_lon = torch.mean((out_fft_lon - target_fft_lon)**2)

        run_loss_run += ((1-lat_lon_bal)*loss_fft_lon + lat_lon_bal*loss_fft_lat)*cw
        totcw+=cw
        
    ## fft_loss_scale included, so the lambda_fft is more intuitive (lambda_fft = .5 --> around half weighted shared between grid and fft loss)
    loss_fft = run_loss_run/totcw*fft_loss_scale
    loss_fft_weighted = lambda_fft*loss_fft
    loss_grid_weighted = ((1-lambda_fft))*loss_grid
    loss = loss_grid_weighted + loss_fft_weighted
    
    # return loss, loss_grid, loss_fft
    return loss

def spectral_sqr_abs2(output, 
                        target, 
                        grid_valid_size = None,
                        wavenum_init_lon = 1, 
                        wavenum_init_lat = 1, 
                        lambda_fft = .5,
                        lat_lon_bal = .5,
                        channels = "all",
                        fft_loss_scale = 1./110.,
                        return_loss_types = False):
        
    """
    Grid and spectral losses, both with mse
    """

    loss_types = {}

    ## loss from grid space
    if grid_valid_size is None: 
        grid_valid_size =  output[...,[0]].flatten().shape[0]
    
    ## grid loss
    loss_grid = torch.sum((output-target)**2)/(grid_valid_size*len(channels))

    loss_types["loss_grid"] = loss_grid.item()

    run_loss_run = torch.zeros(1).float().cuda()
    
    if channels == "all":
        num_spectral_chs = output.shape[-1]
        channels = [["_",i,1./num_spectral_chs] for i in np.arange(num_spectral_chs)]
    
    totcw = 0
    
    ## for valid periodic fft
    ## times x lat x lon x channels
    output2lat = torch.cat([output, torch.flip(output,[1])],dim=1)
    output2lon = torch.cat([output, torch.flip(output,[2])],dim=2)
    target2lat = torch.cat([target, torch.flip(target,[1])],dim=1)
    target2lon = torch.cat([target, torch.flip(target,[2])],dim=2)
    
    for [cname,c,cw] in channels:
        ## it makes sense for me to take fft differences before...if you take mean, you lose more important differences at the equator?
        ## losses from fft, both lat (index 1) and lon (index 2) directions
        if cw != 0:
            ## lat
            out_fft_lat = torch.abs(torch.fft.rfft(output2lat[:,:,:,c],dim=1))[:,wavenum_init_lon:,:]
            target_fft_lat = torch.abs(torch.fft.rfft(target2lat[:,:,:,c],dim=1))[:,wavenum_init_lon:,:]
            loss_fft_lat = torch.mean((out_fft_lat - target_fft_lat)**2)
            ## lon
            out_fft_lon = torch.abs(torch.fft.rfft(output2lon[:,:,:,c],dim=2))[:,:,wavenum_init_lon:]
            target_fft_lon = torch.abs(torch.fft.rfft(target2lon[:,:,:,c],dim=2))[:,:,wavenum_init_lon:]
            loss_fft_lon = torch.mean((out_fft_lon - target_fft_lon)**2)

            run_loss_run += ((1-lat_lon_bal)*loss_fft_lon + lat_lon_bal*loss_fft_lat)*cw
            totcw+=cw

            loss_types[f"loss_fft_lat_{cname}"] = loss_fft_lat.item()
            loss_types[f"loss_fft_lon_{cname}"] = loss_fft_lon.item()
    
    ## fft_loss_scale included, so the lambda_fft is more intuitive (lambda_fft = .5 --> around half weighted shared between grid and fft loss, also considering fft_loss_scale)
    loss_fft = run_loss_run/totcw*fft_loss_scale
    loss_types[f"loss_fft"] = loss_fft.item()
    loss_fft_weighted = lambda_fft*loss_fft
    loss_grid_weighted = ((1-lambda_fft))*loss_grid
    loss = loss_grid_weighted + loss_fft_weighted
    
    # print(loss_types)
    if return_loss_types:
        return loss, loss_types
    else:
        return loss

## jacobian loss
def compute_vjp_batch(model, t_1, t_2):
    output, vjp_func = torch.func.vjp(model, t_1)
    a = vjp_func(t_2 - output)[0]
    return a

## jacobian loss
def compute_vjp(model, input, target):
    output, vjp_func = torch.func.vjp(net_fc, input)
    dx = target - output
    # dx = dx/dx.norm() 
    return vjp_func(dx)[0]

def spectral_sqr_abs2_jc(output, 
                        target,
                        model, 
                        grid_valid_size = None,
                        wavenum_init_lon = 1, 
                        wavenum_init_lat = 1, 
                        lambda_fft = .5,
                        lat_lon_bal = .5,
                        channels = "all",
                        fft_loss_scale = 1./110.):
        
    """
    Grid and spectral losses, both with mse
    """

    loss_types = {}

    ## loss from grid space
    if grid_valid_size is None: 
        grid_valid_size =  output[...,[0]].flatten().shape[0]
        
    loss_grid = torch.sum((output-target)**2)/(grid_valid_size*len(channels))

    loss_types["loss_grid"] = loss_grid.item()

    run_loss_run = torch.zeros(1).float().cuda()
    
    if channels == "all":
        num_spectral_chs = output.shape[-1]
        channels = [["_",i,1./num_spectral_chs] for i in np.arange(num_spectral_chs)]
    
    totcw = 0
    
    ## for valid periodic fft
    ## times x lat x lon x channels
    output2lat = torch.cat([output, torch.flip(output,[1])],dim=1)
    output2lon = torch.cat([output, torch.flip(output,[2])],dim=2)
    target2lat = torch.cat([target, torch.flip(target,[1])],dim=1)
    target2lon = torch.cat([target, torch.flip(target,[2])],dim=2)
    
    for [cname,c,cw] in channels:
        ## it makes sense for me to take fft differences before...if you take mean, you lose more important differences at the equator?
        ## losses from fft, both lat (index 1) and lon (index 2) directions
        if cw != 0:
            ## lat
            out_fft_lat = torch.abs(torch.fft.rfft(output2lat[:,:,:,c],dim=1))[:,wavenum_init_lon:,:]
            target_fft_lat = torch.abs(torch.fft.rfft(target2lat[:,:,:,c],dim=1))[:,wavenum_init_lon:,:]
            loss_fft_lat = torch.mean((out_fft_lat - target_fft_lat)**2)
            ## lon
            out_fft_lon = torch.abs(torch.fft.rfft(output2lon[:,:,:,c],dim=2))[:,:,wavenum_init_lon:]
            target_fft_lon = torch.abs(torch.fft.rfft(target2lon[:,:,:,c],dim=2))[:,:,wavenum_init_lon:]
            loss_fft_lon = torch.mean((out_fft_lon - target_fft_lon)**2)

            run_loss_run += ((1-lat_lon_bal)*loss_fft_lon + lat_lon_bal*loss_fft_lat)*cw
            totcw+=cw

            loss_types[f"loss_fft_lat_{cname}"] = loss_fft_lat.item()
            loss_types[f"loss_fft_lon_{cname}"] = loss_fft_lon.item()
    
    ## fft_loss_scale included, so the lambda_fft is more intuitive (lambda_fft = .5 --> around half weighted shared between grid and fft loss)
    loss_fft = run_loss_run/totcw*fft_loss_scale
    loss_types[f"loss_fft"] = loss_fft.item()
    loss_fft_weighted = lambda_fft*loss_fft
    loss_grid_weighted = ((1-lambda_fft))*loss_grid
    loss = loss_grid_weighted + loss_fft_weighted
    
    # print(loss_types)
    # return loss, loss_grid, loss_fft
    return loss


def spectral_sqr_lonMean(output, 
                               target, 
                               grid_valid_size = None,
                               wavenum_init_lon = 1, 
                               wavenum_init_lat = 1, 
                               lambda_fft = .5,
                               lat_lon_bal = .5,
                               channels = "all",
                               fft_loss_scale = 1./110.,
                               **kwargs):
        
    """
    Grid and spectral losses, both with mse
    """
    # loss1 = torch.sum((output-target)**2)/ocean_grid + torch.sum((output2-target2)**2)/ocean_grid
    
    ## loss from grid space
    if grid_valid_size is None: 
        grid_valid_size =  output[...,[0]].flatten().shape[0]
        
    loss_grid = torch.sum((output-target)**2)/(grid_valid_size*len(channels))
    # loss_grid = torch.mean((output-target)**2)
    # loss1 = torch.abs((output-tnparget))/ocean_grid

    run_loss_run = torch.zeros(1).float().cuda()
    
    if channels == "all":
        num_spectral_chs = output.shape[-1]
        channels = [["_",i,1./num_spectral_chs] for i in np.arange(num_spectral_chs)]
    
    totcw = 0
    for [cname,c,cw] in channels:
        ## it makes sense for me to take fft differences before...if you take mean, you lose more important differences at the equator?
        ## losses from fft, both lat (index 1) and lon (index 2) directions
        ## lon
        out_fft_lon = torch.abs(torch.fft.rfft(output[:,:,:,c],dim=2))[:,:,wavenum_init_lon:]
        out_fft_lon = torch.mean(out_fft_lon,dim=1)
        target_fft_lon = torch.abs(torch.fft.rfft(target[:,:,:,c],dim=2))[:,:,wavenum_init_lon:]
        target_fft_lon = torch.mean(target_fft_lon,dim=1)
        
        loss_fft_lon = torch.mean((out_fft_lon - target_fft_lon)**2)

        run_loss_run += loss_fft_lon*cw
        totcw += cw
        
    ## fft_loss_scale included, so the lambda_fft is more intuitive (lambda_fft = .5 --> around half weighted shared between grid and fft loss)
    loss_fft = run_loss_run/totcw*fft_loss_scale
    loss_fft_weighted = lambda_fft*loss_fft
    loss_grid_weighted = ((1-lambda_fft))*loss_grid
    loss = loss_grid_weighted + loss_fft_weighted
    
    # return loss, loss_grid, loss_fft
    return loss

def spectral_sqr_phase(output, 
                       target, 
                       grid_valid_size = None,
                       wavenum_init_lon = 1, 
                       wavenum_init_lat = 1, 
                       lambda_fft = .5,
                       lat_lon_bal = .5,
                       channels = "all",
                       fft_loss_scale = 1./110.):
        
    """
    Grid and spectral losses, both with mse
    Takes into account sinusoidal phase as well, as opposed to complex norm
    Channel ordering: [batch, lat, lon, channels]
    """

    ## loss from grid space
    if grid_valid_size is None: 
        grid_valid_size =  output[...,[0]].flatten().shape[0] ## single channel grid size
        
    loss_grid = torch.sum((output-target)**2)/(grid_valid_size*len(channels))

    run_loss_run = torch.zeros(1).float().cuda()
    
    if channels == "all":
        num_spectral_chs = output.shape[-1]
        channels = [["_",i,1./num_spectral_chs] for i in np.arange(num_spectral_chs)]
    
    totcw = 0
    for [cname,c,cw] in channels:
        ## it makes sense for me to take fft differences before...if you take mean, you lose more important differences at the equator?
        ## losses from fft, both lat (index 1) and lon (index 2) directions
        ## lat
        out_fft_lat = torch.fft.rfft(output[:,:,:,c],dim=1)[:,wavenum_init_lon:,:]
        target_fft_lat = torch.fft.rfft(target[:,:,:,c],dim=1)[:,wavenum_init_lon:,:]
        loss_fft_lat = torch.mean(torch.abs(out_fft_lat - target_fft_lat)**2)
        ## lon
        out_fft_lon = torch.fft.rfft(output[:,:,:,c],dim=2)[:,:,wavenum_init_lon:]
        target_fft_lon = torch.fft.rfft(target[:,:,:,c],dim=2)[:,:,wavenum_init_lon:]
        loss_fft_lon = torch.mean(torch.abs(out_fft_lon - target_fft_lon)**2)

        run_loss_run += ((1-lat_lon_bal)*loss_fft_lon + lat_lon_bal*loss_fft_lat)*cw
        totcw+=cw

    ## fft_loss_scale included, so the lambda_fft is more intuitive (lambda_fft = .5 --> around half weighted shared between grid and fft loss)
    loss_fft = run_loss_run/totcw*fft_loss_scale
    loss_fft_weighted = lambda_fft*loss_fft
    loss_grid_weighted = ((1-lambda_fft))*loss_grid
    loss = loss_grid_weighted + loss_fft_weighted
    
    # return loss, loss_grid, loss_fft
    return loss

LOSS_FUNCTIONS = {
                  "mse_loss" : mse_loss,
                  "ocean" : ocean_loss,
                  "spectral_sqr_abs" : spectral_sqr_abs,
                  "spectral_sqr_abs2" : spectral_sqr_abs2,
                  "spectral_sqr_phase" : spectral_sqr_phase,
                  "spectral_sqr_lonMean" : spectral_sqr_lonMean,
                  "spectral_sqr_abs2_jc" : spectral_sqr_abs2_jc,
                 }