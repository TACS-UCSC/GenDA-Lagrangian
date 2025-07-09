import torch 

def spectral_sqr_abs2(
    output, 
    target, 
    grid_valid_size=None,
    wavenum_init_lon=1, 
    wavenum_init_lat=50, #change from 1 to 50
    lambda_fft=0.7,
    lat_lon_bal=0.5,
    channels="all",
    fft_loss_scale=1./110.
):
    """
    Grid and spectral losses, both with MSE.
    Modified to accommodate data shape [batch_size, num_channels, latitude_size, longitude_size].
    """
    #For FNO I need to change [batch, lat, long, num_channel] to [batch, num_channed, lat, long]
    output = output.permute(0, 3, 1, 2)
    target  =  target.permute(0, 3, 1, 2)
    # print("outputloss", output.shape, target.shape)

    # Ensure that output and target have the same shape
    assert output.shape == target.shape, "Output and target must have the same shape."

    # Number of channels (second dimension)
    num_channels = output.shape[1]

    # Calculate grid_valid_size if not provided
    if grid_valid_size is None: 
        grid_valid_size = output.numel()  # Total number of elements

    # Compute grid space loss (MSE)
    loss_grid = torch.sum((output - target) ** 2) / (grid_valid_size * num_channels)

    # Compute grid space loss (MAE)
    # Keep MSE 
    # loss_grid = torch.sum(torch.abs(output - target)) / (grid_valid_size * num_channels)

    # Initialize spectral loss accumulator
    run_loss_run = torch.zeros(1, device=output.device, dtype=output.dtype)
    
    # Define channels and their weights
    if channels == "all":
        num_spectral_chs = num_channels
        channels = [["_", i, 1. / num_spectral_chs] for i in range(num_spectral_chs)]
    
    totcw = 0  # Total channel weight

    # Prepare data for periodic FFT along latitude and longitude
    # Concatenate along latitude (dimension 2)
    output2lat = torch.cat([output, torch.flip(output, [2])], dim=2)
    target2lat = torch.cat([target, torch.flip(target, [2])], dim=2)
    
    # Concatenate along longitude (dimension 3)
    output2lon = torch.cat([output, torch.flip(output, [3])], dim=3)
    target2lon = torch.cat([target, torch.flip(target, [3])], dim=3)
    
    # Loop over channels
    for [cname, c, cw] in channels:
        if cw != 0:
            # Select the c-th channel
            output_c = output[:, c, :, :]        # Shape: [batch_size, latitude_size, longitude_size]
            target_c = target[:, c, :, :]
            output2lat_c = output2lat[:, c, :, :]
            target2lat_c = target2lat[:, c, :, :]
            output2lon_c = output2lon[:, c, :, :]
            target2lon_c = target2lon[:, c, :, :]

            # Compute FFT along latitude (dimension 1 after channel selection)
            # out_fft_lat = torch.abs(torch.fft.rfft(output2lat_c, dim=1))[:, wavenum_init_lat:, :]
            # target_fft_lat = torch.abs(torch.fft.rfft(target2lat_c, dim=1))[:, wavenum_init_lat:, :]
            out_fft_lat = torch.abs(torch.fft.rfft(output_c, dim=1))[:, wavenum_init_lat:, :]
            target_fft_lat = torch.abs(torch.fft.rfft(target_c, dim=1))[:, wavenum_init_lat:, :]
            #Ashesh: change to MAE
            #Replace MSE loss:
            # loss_fft_lat = torch.mean((out_fft_lat - target_fft_lat) ** 2)
            # MAE loss
            loss_fft_lat = torch.mean(torch.abs(out_fft_lat - target_fft_lat))

            # Compute FFT along longitude (dimension 2 after channel selection)
            # out_fft_lon = torch.abs(torch.fft.rfft(output2lon_c, dim=2))[:, :, wavenum_init_lon:]
            # target_fft_lon = torch.abs(torch.fft.rfft(target2lon_c, dim=2))[:, :, wavenum_init_lon:]
            out_fft_lon = torch.abs(torch.fft.rfft(output_c, dim=2))[:, :, wavenum_init_lon:]
            target_fft_lon = torch.abs(torch.fft.rfft(target_c, dim=2))[:, :, wavenum_init_lon:]
            #Replace MSE loss to MAE: Ashesh: change to MAE
            # loss_fft_lon = torch.mean((out_fft_lon - target_fft_lon) ** 2)
            # MAE loss
            loss_fft_lon = torch.mean(torch.abs(out_fft_lon - target_fft_lon))

            # Accumulate weighted spectral loss
            run_loss_run += ((1 - lat_lon_bal) * loss_fft_lon + lat_lon_bal * loss_fft_lat) * cw
            totcw += cw

    # Normalize and scale spectral loss
    loss_fft = run_loss_run / totcw * fft_loss_scale

    # Combine grid and spectral losses
    loss = (1 - lambda_fft) * loss_grid + lambda_fft * loss_fft

    return loss
