import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from fingerprint_inversion.fingerinv_utils import fingerInv_ldm, verification_generating_ldm
from fingerprint_inversion.utils import image_grid, load_512
from PIL import Image
from torch.cuda.amp import autocast
from torch import inference_mode


# ==== Config ====
device = "cuda:0"
model_id = "./pretrained_models/DeciDiffusion-v1-0"
pruning_rates = [0.05, 0.10, 0.15]
pruning_weight_path = "/mnt/workspace/tenghuan.th/DDPM_fingerprint/attacked_weights/ldm/pruning/Deci/"
quantized_precisions = [torch.float16, torch.bfloat16]
qr_len = '64'
image_path = './images/qr_code_random_' + qr_len + '.png'
save_path = './results/ldm_robustness_' + qr_len + '_Deci/'
prompt_tar = prompt_src = ''
cfg_tar = cfg_src = 0.0
num_diffusion_steps = 20
default_dtype = torch.float16
os.makedirs(save_path, exist_ok=True)

def load_model_deci(model_id, device, torch_dtype=torch.float16):
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id, custom_pipeline=model_id, torch_dtype=torch_dtype
    ).to(device)
    pipeline.unet = pipeline.unet.from_pretrained(
        model_id, subfolder='flexible_unet', torch_dtype=torch_dtype
    ).to(device)
    return pipeline

def save_image(img, suffix, save_path):
    img_grid = image_grid(img)
    img_grid.save(os.path.join(save_path, f'{suffix}.png'))

def prune_unet_for_sd(pipe, rate, pruning_weight_path, torch_dtype=torch.float16):
    model_parameters = pipe.unet.parameters()
    all_params = torch.nn.utils.parameters_to_vector(model_parameters)
    with torch.no_grad():
        all_params_cpu = all_params.abs().cpu()
        sorted_params, _ = torch.sort(all_params_cpu)
        threshold_idx = int(rate * len(sorted_params))
        threshold = sorted_params[threshold_idx]
        for param in pipe.unet.parameters():
            mask = torch.abs(param) < threshold
            param.data[mask] = 0.0
            param.data = param.data.to(torch_dtype)
    os.makedirs(pruning_weight_path, exist_ok=True)
    save_file = os.path.join(pruning_weight_path, f"unet_pruning_{rate}.pth")
    torch.save(pipe.unet.state_dict(), save_file)

def create_single_row_image(image_paths, gap=15, bg_color=(0, 0, 0)):
    images = []
    for p in image_paths:
        try:
            images.append(Image.open(p))
        except Exception as e:
            print(f"Warning: skipping file {p}: {e}")
    if len(images) == 0:
        print("No valid images for combination.")
        return None
    img_height = images[0].size[1]
    total_width = sum(img.size[0] for img in images) + (len(images) - 1) * gap
    new_image = Image.new('RGB', (total_width, img_height), bg_color)
    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.size[0] + gap
    return new_image

if __name__ == "__main__":
    offsets = (0, 0, 0, 0)
    x0 = load_512(image_path, *offsets, device).to(device, dtype=default_dtype)
    ldm = load_model_deci(model_id, device, torch_dtype=default_dtype)
    scheduler = DDIMScheduler.from_config(
        "./pretrained_models/stable-diffusion-v1-4", subfolder="scheduler"
    )
    ldm.scheduler = scheduler
    ldm.scheduler.set_timesteps(num_diffusion_steps)

    w0 = (ldm.vae.encode(x0).latent_dist.mode() * 0.18215).float().to(device, dtype=default_dtype)
    _, zs, wts = fingerInv_ldm(
        ldm, w0, etas=1.0, prompt=prompt_src, cfg_scale=cfg_src,
        prog_bar=True, num_inference_steps=num_diffusion_steps, dtype=default_dtype
    )
    # zs_tensor = torch.stack(zs)
    zs_tensor = zs

    # Baseline
    w0_baseline, _ = verification_generating_ldm(
        ldm, xT=wts[num_diffusion_steps], etas=1.0,
        prompts=[prompt_tar], cfg_scales=[cfg_tar],
        prog_bar=True, zs=zs_tensor[:num_diffusion_steps], controller=None, dtype=default_dtype
    )
    x0_baseline = ldm.vae.decode(1 / 0.18215 * w0_baseline).sample
    save_image(x0_baseline, "original", save_path)

    # Pruning
    for pruning_rate in pruning_rates:
        pruned_weight_file = os.path.join(pruning_weight_path, f"unet_pruning_{pruning_rate}.pth")
        if not os.path.exists(pruned_weight_file):
            prune_unet_for_sd(ldm, pruning_rate, pruning_weight_path, torch_dtype=default_dtype)
        ldm.unet.load_state_dict(torch.load(pruned_weight_file, map_location=device))
        w0_pruned, _ = verification_generating_ldm(
            ldm, xT=wts[num_diffusion_steps], etas=1.0,
            prompts=[prompt_tar], cfg_scales=[cfg_tar],
            prog_bar=True, zs=zs_tensor[:num_diffusion_steps], controller=None, dtype=default_dtype
        )
        x0_pruned = ldm.vae.decode(1 / 0.18215 * w0_pruned).sample
        save_image(x0_pruned, f"pruned_{pruning_rate}", save_path)

    # Quantization
    for precision in quantized_precisions:
        ldm_quant = load_model_deci(model_id, device, torch_dtype=precision)
        ldm_quant.scheduler = scheduler
        ldm_quant.scheduler.set_timesteps(num_diffusion_steps)
        # with autocast(dtype=torch.float32), inference_mode():
        with autocast(dtype=default_dtype), inference_mode():
            w0_quant, _ = verification_generating_ldm(
                ldm_quant, xT=wts[num_diffusion_steps], etas=1.0,
                prompts=[prompt_tar], cfg_scales=[cfg_tar],
                prog_bar=True, zs=zs[:num_diffusion_steps], controller=None, dtype=precision
            )
            x0_quant = ldm_quant.vae.decode(1 / 0.18215 * w0_quant).sample
        save_image(x0_quant.float(), f"quantized_{precision}", save_path)

    # Combine images
    suffix_list = ["original"] + \
                  [f"pruned_{rate}" for rate in pruning_rates] + \
                  [f"quantized_{p}" for p in quantized_precisions]
    all_image_paths = [os.path.join(save_path, s + '.png') for s in suffix_list if os.path.exists(os.path.join(save_path, s + '.png'))]
    combined_img = create_single_row_image(all_image_paths, gap=15, bg_color=(0, 0, 0))
    if combined_img:
        combined_img.save(os.path.join(save_path, 'combined_image_row.png'))
        print(f"All results are saved into: {os.path.join(save_path, 'combined_image_row.png')}")
    else:
        print("No images generated for combination.")
