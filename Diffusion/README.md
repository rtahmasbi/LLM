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
- CLIPs: OpenCLIP-ViT/G, CLIP-ViT/L, context length 77 tokens
- T5: T5-xxl, context length 77/256 tokens at different stages of training


## CLIP
CLIP stands for Contrastive Language–Image Pre-training — a model developed by OpenAI in 2021.
- It learns to connect text and images in a shared embedding space.
- It provides semantic guidance and conditioning.
- CLIP is the bridge between language and vision in diffusion:


CLIP consists of two encoders:
- Text encoder — turns a text prompt (like “a red cat on a sofa”) into a text embedding vector.
- Image encoder — turns an image into an image embedding vector.


