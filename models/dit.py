# Diffusion Transformer Model

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
References:
1. Building a Vision Transformer Model From Scratch, Matt Nguyen, https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6 

2. Diffusion Transformers: The New Backbone of Generative Vision, Yashas Donthi, https://yashasdonthi.medium.com/diffusion-transformers-the-new-backbone-of-generative-vision-78eb9df657d5
"""

class PatchEmbed(nn.Module):
    """
    Image (pixel) or latent --> to sequence of patch embeddings

    Input: x [B, C, H, W]
    Output: tokens [B, N, d_model] where N = (H/patch) * (W/patch)
    """

    