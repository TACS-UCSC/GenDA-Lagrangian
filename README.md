# **GenDA-Lagrangian**

**Title:** *Generative Lagrangian Data Assimilation for Ocean Dynamics under Extreme Sparsity*

This repository implements a framework that leverages **denoising diffusion probabilistic models (DDPMs)** conditioned on base models such as **UNet** and **Fourier Neural Operators (FNO)** to reconstruct high-resolution ocean dynamics from extremely sparse Lagrangian observations.

Our method combines **Fourier Neural Operators (FNO)** or **UNet** architectures with **DDPMs**. By conditioning the diffusion model on neural operator predictions, we achieve accurate reconstruction of small-scale features and capture high-wavenumber ocean dynamics even under extreme sparsity levels of **99–99.9%**. The framework is validated on benchmark systems, synthetic reanalysis data, and real satellite observations, demonstrating significant improvements over both traditional data assimilation approaches and conventional deep learning models.



## **Data Preparation**

Three different datasets used in this study are available [here on Zenodo](https://zenodo.org/records/15844262). More information about each system or dataset is explained in our paper, section 2, Methods. https://arxiv.org/abs/2507.06479

## **Training Steps**

1. **Train the base models** (either FNO or UNet). The input for these models is the extremely sparse observational data, and the target (label) is the corresponding full-resolution ocean state. Save the trained base model after completion.

2. **Train the DDPM model**, using the previously trained base model as a conditioning input for DDPM.






