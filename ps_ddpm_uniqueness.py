import torch
import os
from torch import autocast, inference_mode
from diffusers import DDPMPipeline, DDIMScheduler
from fingerprint_inversion.utils import image_grid, load_256
from fingerprint_inversion.fingerinv_utils import fingerInv_psddpm, verification_generating_psddpm
import time, calendar
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def load_model(model_id, device, num_diffusion_steps):
    try:
        ddpm = DDPMPipeline.from_pretrained(model_id).to(device)
        scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
        ddpm.scheduler = scheduler
        ddpm.scheduler.set_timesteps(num_diffusion_steps)
        return ddpm
    except Exception as e:
        print(f"Error loading model from {model_id}: {e}")
        return None


model_path = './pretrained_models/ps_ddpm/'

model_ids = [
    "ddpm-ema-bedroom-256",
    "ddpm-ema-cat-256",
    "ddpm-ema-celebahq-256",
    "ddpm-ema-church-256"
]

model_ids_name = [
    "bedroom",
    "cat",
    "celebahq",
    "church"
]

device = "cuda:0"  
num_diffusion_steps = 20

# copyright image path
qr_len = '64'
image_path = './images/qr_code_random_'+qr_len+'.png' 
# qr_len = 'laion_img'
# image_path = './images/laion_art_car_photopartid-0-rowid-2203.png'  

offsets = (0, 0, 0, 0)  
w0 = load_256(image_path, *offsets, device)

# saving path
save_path = './results/ps_ddpms_uniqueness_' + qr_len + '/'
os.makedirs(save_path, exist_ok=True)


# fingerInv for each models
for model_id in model_ids:
    ddpm = load_model(model_path + model_id, device,num_diffusion_steps) 

    with autocast("cuda"):
        _, zs, wts = fingerInv_psddpm(
            ddpm, w0, etas=1, prog_bar=True, num_inference_steps=num_diffusion_steps)
    
    # cross generating
    for j, model_id_2 in enumerate(model_ids):
        ddpm = load_model(model_path + model_id_2, device, num_diffusion_steps)
        with autocast("cuda"):
            generated_image, _ = verification_generating_psddpm(ddpm, xT=wts[num_diffusion_steps], etas=1, prog_bar=True, zs=zs[:(num_diffusion_steps)])


        img_name = f"{model_id}_to_{model_id_2}.png"
        
        img_grid = image_grid(generated_image)
        img_grid.save(os.path.join(save_path, img_name))

print("All sub-images saved.")

# creating a whole image for visualization
# Create a new image (4x4). Assume the image size is 256x256 pixels, with a spacing of 10 pixels.
image_block_size = 256  # size for each image block
spacing = 5  # spacing pixels between image blocks
big_image_size = (4 * image_block_size + 3 * spacing, 4 * image_block_size + 3 * spacing)  
big_image = Image.new('RGB', big_image_size, (0, 0, 0))  # background color (white)

for i in range(4):
    for j in range(4):
        filename = f"{model_ids[i]}_to_{model_ids[j]}.png"
        img_path = os.path.join(save_path, filename)  
        img = Image.open(img_path)

        # calculating pasting position
        x_position = j * (image_block_size + spacing)
        y_position = i * (image_block_size + spacing)
        
        big_image.paste(img, (x_position, y_position))

# saving for png format
big_image.save(os.path.join(save_path, "combined_image_"+qr_len+".png"))
