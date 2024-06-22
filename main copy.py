# pip install accelerate transformers safetensors diffusers

import torch
import numpy as np
import os
import random
from PIL import Image

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
from diffusers.utils import load_image

controlnetModel_path = "D:\IA\Pepsi-ia\IA\models\controlnet-depth-sdxl-1.0"
sd_xl_path = "D:\IA\Pepsi-ia\IA\models\stable-diffusion-xl-base-1.0\sd_xl_base_1.0.safetensors"
reference_image_directory = "D:\IA\Pepsi-ia\IA\Imagenes_de_referencia"
output_directory = "D:\\IA\\Pepsi-ia\\IA\\resultados"
base_prompt = "wooden table with a plate containing ({}) and a generic can of soda as drink"

depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
controlnet = ControlNetModel.from_pretrained(
    controlnetModel_path,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_single_file("D:\IA\Pepsi-ia\IA\models\stable-diffusion-xl-base-1.0\sd_xl_base_1.0_0.9vae.safetensors", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
    sd_xl_path,
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

def select_random_image(directory):
    files = os.listdir(directory)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise ValueError("No image files found in the directory.")
    selected_image = random.choice(image_files)
    return os.path.join(directory, selected_image), selected_image

def get_next_filename(directory):
    existing_files = os.listdir(directory)
    numbers = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]
    next_number = max(numbers, default=0) + 1
    return os.path.join(directory, f"{next_number}.png")

image_path, selected_filename = select_random_image(reference_image_directory)
# print(f"Selected image: {selected_filename}")
image = load_image(image_path).resize((1024, 1024))
controlnet_conditioning_scale = 0.8
depth_image = get_depth_map(image)

user_input = input("Enter the food items (e.g., hamburger and noodles): ")
prompt = base_prompt.format(user_input)
images = pipe(
    prompt,
    image=image,
    control_image=depth_image,
    strength=0.99,
    num_inference_steps=50,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
).images
output_path = get_next_filename(output_directory)
images[0].save(output_path)
print(f"Saved generated image to: {output_path}")