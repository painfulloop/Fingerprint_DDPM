import torch
import os
from torch import autocast, inference_mode
from diffusers import DDPMPipeline, DDIMScheduler
from fingerprint_inversion.utils import image_grid, load_256
from fingerprint_inversion.fingerinv_utils import fingerInv_psddpm, verification_generating_psddpm
import time, calendar
from PIL import Image
import matplotlib.pyplot as plt


def load_model(model_id, device, num_diffusion_steps, data_type = torch.float32):
    try:
        ddpm = DDPMPipeline.from_pretrained(model_id, torch_dtype=data_type).to(device)
        scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
        ddpm.scheduler = scheduler
        ddpm.scheduler.set_timesteps(num_diffusion_steps)
        return ddpm
    except Exception as e:
        print(f"Error loading model from {model_id}: {e}")
        return None

def prune_unet_for_ps_ddpm(pipe, rate, pruning_weight_path):
    model_parameters = pipe.unet.parameters()
    all_params = torch.nn.utils.parameters_to_vector(model_parameters)
    with torch.no_grad():
        all_params_cpu = all_params.abs().to('cpu').sort().values
        threshold_idx = int(rate * len(all_params_cpu))
        threshold = all_params_cpu[threshold_idx]
    with torch.no_grad():
        for param in pipe.unet.parameters():
            mask = torch.abs(param) < threshold
            param.data[mask] = 0.0
    modified_unet_state_dict = pipe.unet.state_dict()
    save_path = os.path.join(pruning_weight_path, f"unet_pruning_{rate}.pth")
    torch.save(modified_unet_state_dict, save_path)


model_path = './pretrained_models/ps_ddpm/'
# model path
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

offsets = (0, 0, 0, 0)  
w0 = load_256(image_path, *offsets, device)

save_path = './results/ps_ddpm_robustness_' + qr_len + '/'
os.makedirs(save_path, exist_ok=True)  

pruning_weight_path_prefix = "./attacked_weights/psddpm/pruning/"


pruning_rates = [0.02,0.05,0.1]
quan_types = [torch.float16, torch.bfloat16]
finetune_iters = [300, 600, 1000]
finetuen_weight_path_prefix = './attacked_weights/psddpm/finetuning/'
finetuen_weight_path_postfix = '_ft_step_'

for model_id in model_ids:
    ddpm = load_model(model_path + model_id, device,num_diffusion_steps, data_type = torch.float32)  

    with autocast("cuda"):
        _, zs, wts = fingerInv_psddpm(
            ddpm, w0, etas=1, prog_bar=True, num_inference_steps=num_diffusion_steps)
    with autocast("cuda"):
        generated_image, _ = verification_generating_psddpm(ddpm, xT=wts[num_diffusion_steps], etas=1, prog_bar=True, zs=zs[:(num_diffusion_steps)])
    img_name = f"{model_id}_org.png"
    img_grid = image_grid(generated_image)
    img_grid.save(os.path.join(save_path, img_name))

    # finetuning attack
    for finetune_iter in finetune_iters:
        ddpm = load_model(finetuen_weight_path_prefix + model_id + finetuen_weight_path_postfix + str(finetune_iter), device,num_diffusion_steps, data_type = torch.float32)  # 加载finetune模型
        with autocast("cuda"):
            generated_image, _ = verification_generating_psddpm(ddpm, xT=wts[num_diffusion_steps], etas=1, prog_bar=True, zs=zs[:(num_diffusion_steps)])
        img_name = f"{model_id}_ft_step_{finetune_iter}.png"
        img_grid = image_grid(generated_image)
        img_grid.save(os.path.join(save_path, img_name))

    # pruning attack
    for pruning_rate in pruning_rates:
        ddpm = load_model(model_path + model_id, device,num_diffusion_steps, data_type = torch.float32) 
        pruning_weight_path = pruning_weight_path_prefix + model_id + '/'
        os.makedirs(pruning_weight_path, exist_ok=True)  

        if not os.path.exists(os.path.join(pruning_weight_path, f"unet_pruning_{pruning_rate}.pth")):
            prune_unet_for_ps_ddpm(ddpm, pruning_rate, pruning_weight_path)
        modified_unet_state_dict = torch.load(os.path.join(pruning_weight_path, f"unet_pruning_{pruning_rate}.pth"))
        ddpm.unet.load_state_dict(modified_unet_state_dict)
        with autocast("cuda"):
            generated_image, _ = verification_generating_psddpm(ddpm, xT=wts[num_diffusion_steps], etas=1, prog_bar=True, zs=zs[:(num_diffusion_steps)])
        img_name = f"{model_id}_pruning_{pruning_rate}.png"
        img_grid = image_grid(generated_image)
        img_grid.save(os.path.join(save_path, img_name))
    
    # quantization
    for quan_type in quan_types:
        ddpm = load_model(model_path + model_id, device,num_diffusion_steps, data_type = quan_type) 
        with autocast("cuda"):
            # , inference_mode()
            generated_image, _ = verification_generating_psddpm(ddpm, xT=wts[num_diffusion_steps], etas=1, prog_bar=True, zs=zs[:(num_diffusion_steps)])
        img_name = f"{model_id}_quantization_{quan_type}.png"
        img_grid = image_grid(generated_image)
        # .convert('L').point(lambda x: 255 if x > 127 else 0)
        img_grid.save(os.path.join(save_path, img_name))

# drawing images
image_width = 256
image_height = 256

# Set image interval and background color
interval = 5  # Interval between images
background_color = (0, 0, 0)  # Background color set to black

# Calculate the number of images to display for each model
num_finetune = len(finetune_iters)
num_pruning = len(pruning_rates)
num_quantization = len(quan_types)

# Number of images displayed per row
images_per_row = 1 + num_finetune + num_pruning + num_quantization

# Calculate the total number of rows
num_rows = len(model_ids)

# Create a large image that combines all images
combined_width = images_per_row * image_width + (images_per_row - 1) * interval
combined_height = num_rows * image_height + (num_rows - 1) * interval
combined_image = Image.new('RGB', (combined_width, combined_height), background_color)

# Create sub-images to separately save the merged images for each attack type
combined_finetune = Image.new('RGB', (num_finetune * image_width + (num_finetune - 1) * interval, num_rows * image_height + (num_rows - 1) * interval), background_color)
combined_pruning = Image.new('RGB', (num_pruning * image_width + (num_pruning - 1) * interval, num_rows * image_height + (num_rows - 1) * interval), background_color)
combined_quantization = Image.new('RGB', (num_quantization * image_width + (num_quantization - 1) * interval, num_rows * image_height + (num_rows - 1) * interval), background_color)

# Fill into the combined image
for i, model_id in enumerate(model_ids):
    # Original model image
    orig_img_path = os.path.join(save_path, f"{model_id}_org.png")
    orig_img = Image.open(orig_img_path)
    combined_image.paste(orig_img, (0, i * (image_height + interval)))  # Paste original image

    # Finetuned model images
    for j, finetune_iter in enumerate(finetune_iters):
        finetuned_img_path = os.path.join(save_path, f"{model_id}_ft_step_{finetune_iter}.png")
        img_finetuned = Image.open(finetuned_img_path)
        x_position = (j + 1) * (image_width + interval)  # Adjust position to ensure interval
        combined_image.paste(img_finetuned, (x_position, i * (image_height + interval)))  # Paste finetuned image
        
        # Also add to finetune combined image
        combined_finetune.paste(img_finetuned, (j * (image_width + interval), i * (image_height + interval)))  # Paste to finetune combination

    # Pruned model images
    for j, pruning_rate in enumerate(pruning_rates):
        pruned_img_path = os.path.join(save_path, f"{model_id}_pruning_{pruning_rate}.png")
        img_pruned = Image.open(pruned_img_path)
        x_position = (num_finetune + 1 + j) * (image_width + interval)  # Adjust position
        combined_image.paste(img_pruned, (x_position, i * (image_height + interval)))  # Paste pruned image
        
        # Also add to pruning combined image
        combined_pruning.paste(img_pruned, (j * (image_width + interval), i * (image_height + interval)))  # Paste to pruning combination

    # Quantized model images
    for j, quan_type in enumerate(quan_types):
        quantized_img_path = os.path.join(save_path, f"{model_id}_quantization_{quan_type}.png")
        img_quantized = Image.open(quantized_img_path)
        x_position = (num_finetune + num_pruning + 1 + j) * (image_width + interval)  # Adjust position
        combined_image.paste(img_quantized, (x_position, i * (image_height + interval)))  # Paste quantized image
        
        # Also add to quantization combined image
        combined_quantization.paste(img_quantized, (j * (image_width + interval), i * (image_height + interval)))  # Paste to quantization combination

# Crop the combined image to remove white margins
bbox = combined_image.getbbox()
if bbox:
    combined_image = combined_image.crop(bbox)

# Save the final combined image
combined_image_path = os.path.join(save_path, 'combined_image.png')  # Path to final PNG file
combined_image.save(combined_image_path)

# Crop and save the finetune combined image
bbox_finetune = combined_finetune.getbbox()
if bbox_finetune:
    combined_finetune = combined_finetune.crop(bbox_finetune)
combined_finetune_path = os.path.join(save_path, 'combined_finetune.png')
combined_finetune.save(combined_finetune_path)

# Crop and save the pruning combined image
bbox_pruning = combined_pruning.getbbox()
if bbox_pruning:
    combined_pruning = combined_pruning.crop(bbox_pruning)
combined_pruning_path = os.path.join(save_path, 'combined_pruning.png')
combined_pruning.save(combined_pruning_path)

# Crop and save the quantization combined image
bbox_quantization = combined_quantization.getbbox()
if bbox_quantization:
    combined_quantization = combined_quantization.crop(bbox_quantization)
combined_quantization_path = os.path.join(save_path, 'combined_quantization.png')
combined_quantization.save(combined_quantization_path)

print(f"All images have been saved:")
print(f"Total combined image: {combined_image_path}")
print(f"Finetune combined image: {combined_finetune_path}")
print(f"Pruning combined image: {combined_pruning_path}")
print(f"Quantization combined image: {combined_quantization_path}")
