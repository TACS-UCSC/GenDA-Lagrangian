# GenDA-Lagrangian

Title: Generative Lagrangian data assimilation for ocean dynamics under extreme sparsity

This repository implements a DDPM conditioned to base models (UNET and FNO) to reconstruct high-resolution ocean dynamics from extremely sparse Lagrangian observations. 

Our framework combines Fourier neural operators (FNO)/UNET with denoising diffusion probabilistic models (DDPMs). By conditioning the diffusion model on neural operator predictions, we reconstruct small-scale features, and captures high-wavenumber ocean dynamics even at extreme sparsity levels (up to 99–99.9%). We validate our method on benchmark systems, synthetic reanalysis data, and real satellite observations, demonstrating significant improvements over both traditional data assimilation models and conventional neural networks. 




