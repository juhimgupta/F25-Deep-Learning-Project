# IDL25Fall-HW5

# Files Walkthrough

## Denoising Diffusion Probabilistic Models (DDPM)-Related Files**
```
1. pipelines/ddpm.py
2. schedulers/scheduling_ddpm.py
3. train.py
4. configs/ddpm.yaml
```
### **DDPM Pipeline: Inference and Image Generation**
The **`ddpm.py`** script defines the **DDPMPipeline** class, which is responsible for generating images from random noise using the **DDPM** framework. Key components of this pipeline include:

- **Unet (Noise Prediction Network)**: Neural network used to predict the noise at each timestep during the reverse diffusion process
- **Scheduler**: Defines how the noise evolves through each timestep in both the forward and reverse diffusion processes
- **VAE (Variational Autoencoder)**: Used in **latent DDPM**, allowing the model to operate in a compressed latent space for more efficient inference
- **Class Embedder**: Used for **conditional image generation** via **Classifier-Free Guidance (CFG)**, enabling the generation of class-specific images

The script also contains helper functions like:
- **`numpy_to_pil`**: Converts numpy arrays of generated images to PIL format
- **`progress_bar`**: Displays a progress bar to track the status of image generation during inference

The main **`__call__`** method performs the **inference** process, where random noise is progressively denoised step-by-step to generate high-quality images.