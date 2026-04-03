# Diffusion
## Example
```py
from diffusers import StableDiffusionPipeline

model_id = "sd-dreambooth-library/mr-potato-head"
# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)


prompt = "an abstract oil painting of sks mr potato head by picasso"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image

```


# Stable Diffusion

https://huggingface.co/stabilityai/stable-diffusion-3.5-large



```py
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("capybara.png")


#####
import torch
from diffusers import DiffusionPipeline

# switch to "mps" for apple devices
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", dtype=torch.bfloat16, device_map="cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]

```


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

