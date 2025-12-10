################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch
from torchvision.utils import make_grid
import numpy as np


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Sample epsilon from standard normal distribution N(0, 1)
    epsilon = torch.randn_like(mean)
    # Apply reparameterization trick: z = std * epsilon + mean
    z = std * epsilon + mean
    #######################
    # END OF YOUR CODE    #
    #######################
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # KL divergence formula: 0.5 * Σ_d [exp(2 log σ) + μ² - 1 - 2 log σ]
    # incorprating the funtionsw params yields: 0.5 * Σ_d [exp(2 * log_std) + mean² - 1 - 2 * log_std]
    two_log_std = 2 * log_std
    KLD = 0.5 * torch.sum(torch.exp(two_log_std) + mean.pow(2) - 1 - two_log_std, dim=-1)
    #######################
    # END OF YOUR CODE    #
    #######################
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    dims = img_shape[1:]  # [channels, height, width]
    product_dims = torch.prod(torch.tensor(dims, dtype=torch.float32))
    log2_e = torch.log2(torch.e)
    # Compute bits per dimension- Formula: bpd = nll · log₂(e) · (∏ᵢ dᵢ)⁻¹
    bpd = elbo * log2_e / product_dims
    #######################
    # END OF YOUR CODE    #
    #######################
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """

    ## Hints:
    # - You can use the icdf method of the torch normal distribution  to obtain z values at percentiles.
    # - Use the range [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a softmax after the decoder

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Get device from decoder
    device = decoder.device
    
    # Infer z_dim from decoder by checking the first linear layer's input features
    z_dim = None
    for name, module in decoder.named_modules():
        if isinstance(module, torch.nn.Linear):
            z_dim = module.in_features
            break
    
    # If not found, try to infer from first parameter shape
    if z_dim is None:
        for name, param in decoder.named_parameters():
            if len(param.shape) == 2:
                z_dim = param.shape[1]
                break
    
    # Fallback: try a dummy forward pass
    if z_dim is None:
        for test_dim in [2, 10, 20, 32, 64]:
            try:
                dummy_z = torch.randn(1, test_dim, device=device)
                with torch.no_grad():
                    _ = decoder(dummy_z)
                z_dim = test_dim
                break
            except:
                continue
    
    if z_dim is None:
        raise ValueError("Could not infer z_dim from decoder")
    
    # Create percentiles: [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size]
    percentiles = torch.linspace(0.5 / grid_size, (grid_size - 0.5) / grid_size, grid_size, device=device)
    
    # Use icdf of standard normal distribution to convert percentiles to z values
    normal_dist = torch.distributions.Normal(0, 1)
    z_values = normal_dist.icdf(percentiles)
    
    # Create meshgrid for 2D latent space
    z1, z2 = torch.meshgrid(z_values, z_values, indexing='ij')
    
    # Flatten the grid to create batch of z vectors
    z1_flat = z1.flatten()  # Shape: [grid_size**2]
    z2_flat = z2.flatten()   # Shape: [grid_size**2]
    
    # Create z vectors: first 2 dims from meshgrid, rest set to 0
    z_batch = torch.zeros(grid_size**2, z_dim, device=device)
    z_batch[:, 0] = z1_flat
    z_batch[:, 1] = z2_flat
    
    # Pass through decoder
    decoder.eval()
    with torch.no_grad():
        logits = decoder(z_batch)  # Shape: [grid_size**2, num_channels, H, W]
    
    # Apply softmax to get probabilities because decoder outputs logits
    probs = torch.softmax(logits, dim=1)  # Shape: [grid_size**2, num_channels, H, W]
    
    # Convert to expected format for make_grid
    # The images should represent decoder's output means (not binarized samples)
    # For 4-bit images, we have 16 channels representing values 0-15
    # Compute expected value (mean) as weighted sum: Σ (class_value * probability)
    if probs.shape[1] > 1:
        # Create class values tensor: [0, 1, 2, ..., 15]
        class_values = torch.arange(probs.shape[1], dtype=torch.float32, device=device).view(1, -1, 1, 1)
        # Compute expected value: weighted sum of class values by their probabilities
        img_values = (probs * class_values).sum(dim=1) / 15.0  # Shape: [grid_size**2, H, W], normalized to [0, 1]
        # Add channel dimension
        images = img_values.unsqueeze(1)  # Shape: [grid_size**2, 1, H, W]
    else:
        images = probs
    
    # Use make_grid to combine images into a grid
    img_grid = make_grid(images, nrow=grid_size, normalize=True, value_range=(0, 1), pad_value=0.5)
    #######################
    # END OF YOUR CODE    #
    #######################

    return img_grid

