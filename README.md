# **GenDA-Lagrangian**

**Title:** *Generative Lagrangian Data Assimilation for Ocean Dynamics under Extreme Sparsity*

This repository implements a framework that leverages **denoising diffusion probabilistic models (DDPMs)** conditioned on base models such as **UNet** and **Fourier Neural Operators (FNO)** to reconstruct high-resolution ocean dynamics from extremely sparse Lagrangian observations.

Our method combines **Fourier Neural Operators (FNO)** or **UNet** architectures with **DDPMs**. By conditioning the diffusion model on neural operator predictions, we achieve accurate reconstruction of small-scale features and capture high-wavenumber ocean dynamics even under extreme sparsity levels of **99–99.9%**. The framework is validated on benchmark systems, synthetic reanalysis data, and real satellite observations, demonstrating significant improvements over both traditional data assimilation approaches and conventional deep learning models.





