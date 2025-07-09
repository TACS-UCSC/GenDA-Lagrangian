import torch
import numpy as np


print("Load data for 99  sparsity with mask as a seperate channel")
def get_dynamics_data(
    data_dir_mask = "/glade/derecho/scratch/nasefi/compressed_sensing/Beta_channel-turbulence_n/data_directories/processed_data/New_norm_masked/channel_masked/ch_mask99.npy",
    data_dir_original = "/glade/derecho/scratch/nasefi/compressed_sensing/Beta_channel-turbulence_n/data_directories/processed_data/New_norm_original/chorigin_all.npy", 

):
   
    with open(data_dir_mask, 'rb') as f:
        data_masked = np.load(f) 
    with open(data_dir_original, 'rb') as f:
        data_original = np.load(f)

  
    
    print("data_masked", data_masked.shape)            
    print("data_original", data_original.shape)         
    

    u_train_m = torch.Tensor(data_masked[:7000])
    u_train_o = torch.Tensor(data_original[:7000])
    u_test_m = torch.Tensor(data_masked[7000:])
    u_test_o = torch.Tensor(data_original[7000:]) 

    
    print("Curious", type(u_train_m.shape), u_train_m.shape)   
    print("curiousss", u_train_o.shape)
    print("curiousnn", u_test_m.shape)     
    print("curiousddd", u_test_o.shape)
    
    u_train_m = u_train_m.permute(0,2, 3, 1)
    u_train_o = u_train_o.permute(0,2, 3, 1)
    u_test_m = u_test_m.permute(0,2, 3, 1)
    u_test_o = u_test_o.permute(0,2, 3, 1)

    print("permute", type(u_train_m.shape), u_train_m.shape)
      


    return u_train_m, u_train_o, u_test_m, u_test_o



print("Finish")

u_train_m, u_train_o, u_test_m, u_test_o = get_dynamics_data()
 

print("FNew_u_train_m", u_train_m.shape)    
print("FNew_u_train_0", u_train_o.shape)
print("FNew_u_test_m", u_test_m.shape)    
print("FNew_u_test_o", u_test_o.shape)
      

# #  input shape: (batchsize, x=256, y=256, c=1)
#         output: the solution of the next timestep
#         output shape: (batchsize, x=64, y=64, c=1)
