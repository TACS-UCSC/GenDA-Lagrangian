
import torch
import random
import numpy as np

#data_dir_mask = "/glade/derecho/scratch/nasefi/compressed_sensing/Beta_channel-turbulence_n/data_directories/turb2d_denorm/new99.npy",

def set_seed(seed=33):
    """Set all seeds for the experiments.
    Args:
        seed (int, optional): seed for pseudo-random generated numbers.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

print("Load data for 90 sparsity")
def get_dynamics_data(
    data_dir_mask = "/glade/derecho/scratch/nasefi/compressed_sensing/Beta_channel-turbulence_n/data_directories/turb2d_denorm/new90.npy",
    data_dir_original = "/glade/derecho/scratch/nasefi/compressed_sensing/Beta_channel-turbulence_n/data_directories/turb2d_denorm/origin_norm.npy", 
):
    """Get training and test data as well as associated coordinates, depending on the dataset name.
    Args:
        data_dir (str): path to the dataset directory
        dataset_name (str): dataset name (e.g. "navier-stokes)
        ntrain (int): number of training samples
        ntest (int): number of test samples
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_tr]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_tr*len(x)). Defaults to 1.
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_te]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_te*len(x)). Defaults to 1.
        same_grid (bool, optional): If True, all the trajectories avec the same grids.
    Raises:
        NotImplementedError: _description_
    Returns:
        u_train (torch.Tensor): (ntrain, ..., T)
        u_test (torch.Tensor): (ntest, ..., T)
        grid_tr (torch.Tensor): coordinates of u_train
        grid_te (torch.Tensor): coordinates of u_test
    """

    with open(data_dir_mask, 'rb') as f:
        data_masked = np.load(f) 
    with open(data_dir_original, 'rb') as f:
        data_original = np.load(f)

    print("masked", data_masked.shape)
    print("original",type(data_original), data_original.shape)


    u_train_m = torch.Tensor(data_masked[:7000])
    u_train_o = torch.Tensor(data_original[:7000])
    u_test_m = torch.Tensor(data_masked[7000:])
    u_test_o = torch.Tensor(data_original[7000:]) 


    print("Curious", type(u_train_m.shape), u_train_m.shape)   
    print("curiousss", u_train_o.shape)
    print("curiousnn", u_test_m.shape)     
    print("curiousddd", u_test_o.shape)



    return u_train_m, u_train_o, u_test_m, u_test_o



print("Finish")

u_train_m, u_train_o, u_test_m, u_test_o = get_dynamics_data()


print("FNew_u_train_m", u_train_m.shape)   
print("FNew_u_train_0", u_train_o.shape)    
print("FNew_u_test_m", u_test_m.shape)   
print("FNew_u_test_o", u_test_o.shape)

