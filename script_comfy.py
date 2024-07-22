import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from PIL import Image
from io import BytesIO
import numpy as np
import boto3
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)




def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")

def tensor_to_pil(images):
    for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img


add_comfyui_directory_to_sys_path()
add_extra_model_paths()

from nodes import (
    CheckpointLoaderSimple,
    VAEDecode,
    NODE_CLASS_MAPPINGS,
    KSampler,
    LoraLoader,
    CLIPTextEncode,
    EmptyLatentImage,
)

@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.json
    user_input = data.get('prompt', '')
    uuid = data.get('uuid', '')

    if not user_input:
        return jsonify({'error': 'No prompt provided'}), 400
    

    prompt_template = "A magazine photograph of an amazing plate of {} by chef Gordan Ramsey and (one) ((blue pepsi can)) on the side, award winning photograph by an acclaimed photographer, f1.8, cinematic lighting, focused composition lots of detail, extremely detailed, full of detail, wide color range, high dynamics"
    prompt = prompt_template.format(user_input)
    
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="cyberrealisticXL_v20.safetensors"
        )

        loraloader = LoraLoader()
        loraloader_11 = loraloader.load_lora(
            lora_name="blue pepsi can-000005.safetensors",
            strength_model=0.8,
            strength_clip=1,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )
        
        ksampler = KSampler()
        vaedecode = VAEDecode()

        while True:
                emptylatentimage = EmptyLatentImage()
                emptylatentimage_5 = emptylatentimage.generate(
                    width=1024, height=1024, batch_size=1
                )

                prompt = prompt_template.format(user_input)
                cliptextencode = CLIPTextEncode()
                cliptextencode_6 = cliptextencode.encode(
                    text=prompt,
                    clip=get_value_at_index(loraloader_11, 1),
                )

                cliptextencode_7 = cliptextencode.encode(
                    text="cocacola, two cans, overlay, grit, dull, washed out, low contrast, blurry, hazy, malformed, warped, deformed, text, watermark, unfocused background, poorly drawn, bad quality, unappetizing, unrealistic proportions, pixelated, low resolution, bad lighting, multiple plates, low detail, low quality, worst quality, 2d, 3d, cartoon, illustration, painting, sketch, copyright, boring",
                    clip=get_value_at_index(loraloader_11, 1),
                )

                for q in range(1):
                    ksampler_3 = ksampler.sample(
                        seed=random.randint(1, 2**64),
                        steps=30,
                        cfg=7,
                        sampler_name="euler",
                        scheduler="normal",
                        denoise=1,
                        model=get_value_at_index(loraloader_11, 0),
                        positive=get_value_at_index(cliptextencode_6, 0),
                        negative=get_value_at_index(cliptextencode_7, 0),
                        latent_image=get_value_at_index(emptylatentimage_5, 0),
                    )

                    vaedecode_8 = vaedecode.decode(
                        samples=get_value_at_index(ksampler_3, 0),
                        vae=get_value_at_index(checkpointloadersimple_4, 2),
                    )

                    tensor = get_value_at_index(vaedecode_8, 0)
                    image = tensor_to_pil(tensor)
                    image.save('output_image.png')

                    # Save PIL Image as PNG in Memory
                    buffer = BytesIO()
                    image.save(buffer, format="PNG")
                    buffer.seek(0)

                    # Upload to S3
                    s3 = boto3.client('s3')
                    bucket_name = 'elasticbeanstalk-pepsi-ia-1'
                    s3_key = f"images/{uuid}.png"

                    s3.upload_fileobj(buffer, bucket_name, s3_key)

                    return jsonify({'message': 'Image generated and uploaded to S3 successfully', 's3_key': s3_key})
                        


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)