# Diffusion
## Stable Diffusion examples from pretrained
- [stable_diffusion_ex1.py](Diffusion/stable_diffusion_ex1.py)
- [stable_diffusion_ex1.py](Diffusion/stable_diffusion_ex2.py)



## Text Encoders
- **CLIPs**: OpenCLIP-ViT/G, CLIP-ViT/L, context length 77 tokens
- **T5**: T5-xxl, context length 77/256 tokens at different stages of training


## CLIP - Contrastive Language–Image Pre-training
CLIP — a model developed by OpenAI in 2021.
- It learns to connect text and images in a shared embedding space.
- It provides semantic guidance and conditioning.
- CLIP is the bridge between language and vision in diffusion:


CLIP consists of two encoders:
- Text encoder — turns a text prompt (like “a red cat on a sofa”) into a text embedding vector.
- Image encoder — turns an image into an image embedding vector.

[CLIP paper](https://arxiv.org/abs/2103.00020)



# Diffusion from scratch
- [My codes diffusion_from_scratch](Diffusion/diffusion_from_scratch)
- https://github.com/gmongaras/Diffusion_models_from_scratch
- https://github.com/nickd16/Diffusion-Models-from-Scratch/
- https://towardsdatascience.com/diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946/



# papers
- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/pdf/2006.11239). It is Unconditional Image Generation
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672). It is a conditional diffusion model.
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
- [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502)
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)
- [U-net (Convolutional Networks for Biomedical Image Segmentation)](https://arxiv.org/abs/1505.04597)
- [ConvNext (A ConvNet for the 2020s)](https://arxiv.org/abs/2201.03545)



# Main conditioning types
## 1. Class-conditional diffusion
- Condition on labels (e.g., “dog”, “car”)
- Early example: Improved DDPM
- Technique: classifier guidance or classifier-free guidance

## 2. Text-to-image diffusion
- Condition on text prompts
- Popular models:
    - Stable Diffusion
    - DALL·E 2
    - Imagen
👉 These use a text encoder (often CLIP-like) and cross-attention.

## 3. Image-to-image / guided diffusion
- Condition on another image
- Examples:
    - Super-resolution
    - Inpainting
    - Style transfer
- A well-known framework:
    - [Palette](https://iterative-refinement.github.io/palette/)

## 4. Structured conditioning
- Condition on richer signals:
    - Segmentation maps
    - Pose skeletons
    - Depth maps
    - Edge maps
- Example:
    - ControlNet (adds strong spatial conditioning to Stable Diffusion) [ref](https://arxiv.org/pdf/2302.05543)

## 5. Multimodal conditioning
- Text + image + layout
- Scene graphs or 3D constraints
- Example:
    - [GLIDE](https://arxiv.org/pdf/2112.10741) and [github](https://github.com/openai/glide-text2im)

