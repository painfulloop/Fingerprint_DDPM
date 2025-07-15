import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from fingerprint_inversion.fingerinv_utils import fingerInv_ldm, verification_generating_ldm
from fingerprint_inversion.utils import image_grid, load_512
from torch import autocast, inference_mode
from PIL import Image

# ==== Config ====
device = "cuda:0"
model_id = "./pretrained_models/stable-diffusion-v1-4"
pruning_rates = [0.1, 0.3, 0.5]
pruning_weight_path = "./attacked_weights/ldm/pruning/SD1-4/"
finetuned_model_ids = [
    './attacked_weights/ldm/finetuning/Deliberate',
    './attacked_weights/ldm/finetuning/stable-diffusion-v1-5',
    './attacked_weights/ldm/finetuning/Realistic_Vision_V2.0',
    './attacked_weights/ldm/finetuning/anything-v4.0'
]
quantized_precisions = [torch.bfloat16, torch.float16]
qr_len = '64'
image_path = './images/qr_code_random_' + qr_len + '.png'
prompt_src = ''
prompt_tar = ''
cfg_src = 0.0
cfg_tar = 0.0
num_diffusion_steps = 20
skip = 0
eta = 1.0
save_path = './results/ldm_robustness_' + qr_len + '_SD/'
os.makedirs(save_path, exist_ok=True)

# ==== Utils ====
def load_model(model_id, device, torch_dtype=None):
    if torch_dtype:
        return StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
    else:
        return StableDiffusionPipeline.from_pretrained(model_id).to(device)

def save_image(img, suffix, save_path):
    img_grid = image_grid(img)
    img_grid.save(os.path.join(save_path, f'{suffix}.png'))

def prune_unet_for_sd(pipe, rate, pruning_weight_path):
    """Hard prune pipeline's UNet and save state dict."""
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
    os.makedirs(pruning_weight_path, exist_ok=True)
    save_file = os.path.join(pruning_weight_path, f"unet_pruning_{rate}.pth")
    torch.save(pipe.unet.state_dict(), save_file)

def create_single_row_image(image_paths, gap=15, bg_color=(0, 0, 0)):
    """Combine multiple images horizontally into one."""
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

# ==== Main ====
if __name__ == "__main__":
    offsets = (0, 0, 0, 0)
    x0 = load_512(image_path, *offsets, device)
    ldm_stable = load_model(model_id, device)
    scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    ldm_stable.scheduler = scheduler
    ldm_stable.scheduler.set_timesteps(num_diffusion_steps)

    # ==== Encode input image & inversion ====
    with autocast(device), inference_mode():
        w0 = (ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).float()

    _, zs, wts = fingerInv_ldm(
        ldm_stable, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_src,
        prog_bar=True, num_inference_steps=num_diffusion_steps
    )

    # ==== Baseline (original pipeline) ====
    w0_baseline, _ = verification_generating_ldm(
        ldm_stable, xT=wts[num_diffusion_steps-skip], etas=eta,
        prompts=[prompt_tar], cfg_scales=[cfg_tar],
        prog_bar=True, zs=zs[:num_diffusion_steps-skip], controller=None
    )
    with autocast(device), inference_mode():
        x0_baseline = ldm_stable.vae.decode(1 / 0.18215 * w0_baseline).sample
    save_image(x0_baseline, "original", save_path)

    # ==== Pruned UNet inference ====
    for pruning_rate in pruning_rates:
        pruned_weight_file = os.path.join(pruning_weight_path, f"unet_pruning_{pruning_rate}.pth")
        if not os.path.exists(pruned_weight_file):
            prune_unet_for_sd(ldm_stable, pruning_rate, pruning_weight_path)
        ldm_stable.unet.load_state_dict(torch.load(pruned_weight_file, map_location=device))
        w0_pruned, _ = verification_generating_ldm(
            ldm_stable, xT=wts[num_diffusion_steps-skip], etas=eta,
            prompts=[prompt_tar], cfg_scales=[cfg_tar],
            prog_bar=True, zs=zs[:num_diffusion_steps-skip], controller=None
        )
        with autocast(device), inference_mode():
            x0_pruned = ldm_stable.vae.decode(1 / 0.18215 * w0_pruned).sample
        save_image(x0_pruned, f"pruned_{pruning_rate}", save_path)

    # ==== Finetuned UNet models ====
    for ft_model_path in finetuned_model_ids:
        ldm_finetune = load_model(ft_model_path, device)
        ft_scheduler = DDIMScheduler.from_config(ft_model_path, subfolder="scheduler")
        ldm_finetune.scheduler = ft_scheduler
        ldm_finetune.scheduler.set_timesteps(num_diffusion_steps)
        w0_ft, _ = verification_generating_ldm(
            ldm_finetune, xT=wts[num_diffusion_steps-skip], etas=eta,
            prompts=[prompt_tar], cfg_scales=[cfg_tar],
            prog_bar=True, zs=zs[:num_diffusion_steps-skip], controller=None
        )
        with autocast(device), inference_mode():
            x0_ft = ldm_finetune.vae.decode(1 / 0.18215 * w0_ft).sample
        tag = ft_model_path.replace("/", "_")
        save_image(x0_ft, f"finetuned_{tag}", save_path)

    # ==== Quantized UNet inference ====
    for precision in quantized_precisions:
        ldm_quant = load_model(model_id, device, torch_dtype=precision)
        ldm_quant.scheduler = scheduler
        ldm_quant.scheduler.set_timesteps(num_diffusion_steps)
        q_wts = [w.to(device, dtype=precision) for w in wts]
        q_zs = [z.to(device, dtype=precision) for z in zs]
        zs_tensor = torch.stack(q_zs)  

        w0_quant, _ = verification_generating_ldm(
            ldm_quant, xT=q_wts[num_diffusion_steps-skip], etas=eta,
            prompts=[prompt_tar], cfg_scales=[cfg_tar],
            prog_bar=True, zs=zs_tensor[:num_diffusion_steps-skip], controller=None
        )
        with autocast(device), inference_mode():
            x0_quant = ldm_quant.vae.decode(1 / 0.18215 * w0_quant).sample
        save_image(x0_quant, f"quantized_{precision}", save_path)

    # ==== Combine all generated images into a single row ====
    suffix_list = ['original'] + \
                  [f'pruned_{rate}' for rate in pruning_rates] + \
                  [f'finetuned_{path.replace("/", "_")}' for path in finetuned_model_ids] + \
                  [f'quantized_{p}' for p in quantized_precisions]
    all_image_paths = [os.path.join(save_path, s + '.png') for s in suffix_list if os.path.exists(os.path.join(save_path, s + '.png'))]
    combined_img = create_single_row_image(all_image_paths, gap=15, bg_color=(0, 0, 0))
    if combined_img:
        combined_img.save(os.path.join(save_path, 'combined_image_row.png'))
        print(f"All results are saved into: {os.path.join(save_path, 'combined_image_row.png')}")
    else:
        print("No images generated for combination.")
