# pip install accelerate transformers safetensors diffusers

import torch
import numpy as np
from PIL import Image

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
from diffusers.utils import load_image


depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
controlnet = ControlNetModel.from_pretrained(
    "D:\IA\Pepsi-ia\IA\models\controlnet-depth-sdxl-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_single_file("D:\IA\Pepsi-ia\IA\models\stable-diffusion-xl-base-1.0\sd_xl_base_1.0_0.9vae.safetensors", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
    "D:\IA\Pepsi-ia\IA\models\stable-diffusion-xl-base-1.0\sd_xl_base_1.0.safetensors",
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


prompt = "wooden table with a plate containg (hamburguer and noodles) and a generic can of soda as drink"
image = load_image(
    "D:\IA\Pepsi-ia\IA\Imagenes_de_referencia\ComfyUI_00801_.png"
).resize((1024, 1024))
controlnet_conditioning_scale = 0.8  # recommended for good generalization
depth_image = get_depth_map(image)

images = pipe(
    prompt,
    image=image,
    control_image=depth_image,
    strength=0.99,
    num_inference_steps=50,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
).images
images[0].save(f"prueba1.png")