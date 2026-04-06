
# unet
For image segmentation
https://theaisummer.com/unet-architectures/

- [Unet 2015](https://arxiv.org/pdf/1505.04597)
- [UNet++](https://arxiv.org/pdf/1807.10165)


## Time Conditioning (Core Difference)
Classic U-Net:
- Input: image
- Output: segmentation map

Diffusion U-Net:
- Input: noisy image + timestep (t)
- Output: predicted noise (ε) or velocity

## Attention Layers (Not in Original U-Net)
Diffusion U-Nets add:
- Self-attention (spatial relationships)
- Cross-attention (conditioning, e.g., text)

## Residual Blocks Instead of Plain Conv Blocks
Original U-Net:
- Conv → ReLU → Conv
- Diffusion U-Net:
ResNet-style blocks:
- Conv → GroupNorm → SiLU → Conv
- Skip connection inside each block

## Normalization Strategy
Original:
- Often no normalization or simple batch norm
Diffusion:
- GroupNorm (standard choice)

Why?

- Works better for small batch sizes
- More stable for generative training

## Latent Space vs Pixel Space (Modern Twist)
In Stable Diffusion:
- U-Net operates on latent representations, not raw pixels
- Uses a VAE encoder/decoder around it

Benefits:

10–50× cheaper compute
Enables high-resolution generation
