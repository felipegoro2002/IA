import os
import sys
import json
import boto3
import random
import torch
from PIL import Image
from io import BytesIO
import numpy as np
import requests
import urllib.parse

# Set up AWS credentials

sqs = boto3.client('sqs')
s3 = boto3.client('s3')
queue_url = 'https://sqs.us-east-1.amazonaws.com/220959411709/pepsiqueue.fifo'
bucket_name = 'elasticbeanstalk-pepsi-ia-1'

def get_value_at_index(obj, index):
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def find_path(name, path=None):
    if path is None:
        path = os.getcwd()
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name
    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None
    return find_path(name, parent_directory)

def add_comfyui_directory_to_sys_path():
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")

def add_extra_model_paths():
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

with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="cyberrealisticXL_v21.safetensors"
        )

        loraloader = LoraLoader()
        loraloader_11 = loraloader.load_lora(
            lora_name="blue pepsi can-000005.safetensors",
            strength_model=0.9,
            strength_clip=1,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        ksampler = KSampler()
        vaedecode = VAEDecode()

def process_message(message):
    data = json.loads(message['Body'])
    user_input = data['prompt']
    uuid = data['uuid']

    prompt_template = "A magazine photograph of an amazing plate of ((({}))) by chef Gordan Ramsey and (one) unmodified ((blue pepsi can)) on the side, f1.8, cinematic lighting, focused composition lots of detail, extremely detailed, full of detail, wide color range, high dynamics, high resolution, 4k"
    prompt = prompt_template.format(user_input)

    with torch.inference_mode():
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
            text="cocacola, yellow can, orange can, two cans, humans, overlay, grit, dull, washed out, low contrast, blurry, hazy, malformed, warped, deformed, text, watermark, unfocused background, poorly drawn, bad quality, unappetizing, unrealistic proportions, pixelated, low resolution, bad lighting, multiple plates, low detail, low quality, worst quality, cartoon, illustration, painting, sketch, copyright, boring",
            clip=get_value_at_index(loraloader_11, 1),
        )

        ksampler_3 = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=30,
            cfg=9,
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

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        s3_key = f"pruebas/{uuid}.png"
        s3.upload_fileobj(buffer, bucket_name, s3_key, ExtraArgs={'ContentType': 'image/png'})
        
        object_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"

        confirmacion = {
                "titulo": user_input,
                "media": object_url,
                "uuid": uuid
            }
        url = "http://ia-pepsi-env.eba-q2ivk33d.us-east-1.elasticbeanstalk.com/media_upload/"
        headers = {"Content-Type": "application/json"}
        requests.post(url, headers=headers, data=json.dumps(confirmacion))
        print(f"Prompt: {user_input} URL: {object_url}")

def poll_queue():
    while True:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10
        )

        messages = response.get('Messages', [])
        if messages:
            for message in messages:
                process_message(message)
                sqs.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=message['ReceiptHandle']
                )
        else:
            print("No messages in the queue.")

if __name__ == '__main__':
    poll_queue()
