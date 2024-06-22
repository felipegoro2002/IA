from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_single_file(
    "D:\IA\Pepsi-ia\IA\models\stable-diffusion-xl-base-1.0\sd_xl_base_1.0.safetensors", torch_dtype=torch.float16).to("cuda")
prompt = "an astronaut rinding a horse"

image = pipe(
    prompt,
    guidance_scale=7,
    target_size=(1024,1024),
    num_inference_steps=25
    ).images[0]
image.save("ejemplo.png")