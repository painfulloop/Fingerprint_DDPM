import os
import torch
from diffusers import PixArtAlphaPipeline, DDIMScheduler
from fingerprint_inversion.fingerinv_utils import fingerInv_ldm_pixart, verification_generating_ldm_pixart
from fingerprint_inversion.utils import image_grid, load_512
from torch import inference_mode
from torch.cuda.amp import autocast
from PIL import Image

# ==== Config ====
device = "cuda:0"
model_id = "./pretrained_models/pixart"
pruning_rates = [0.05, 0.10, 0.15]
pruning_weight_path = "./attacked_weights/ldm/pruning/pixart/"
quantized_precisions = [torch.float16, torch.bfloat16]
qr_len = '64'
image_path = f'./images/qr_code_random_{qr_len}.png'
save_path = f'./results/ldm_robustness_{qr_len}_Pixart/'
prompt_src = prompt_tar = ''
cfg_src = cfg_tar = 0.0
num_diffusion_steps = 20
default_dtype = torch.float32  
os.makedirs(save_path, exist_ok=True)

def load_model_pixart(model_id, device, torch_dtype=default_dtype):
    pipe = PixArtAlphaPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=True).to(device)
    return pipe

def prune_transformer_for_pixart(pipe, rate, pruning_weight_path):
    model_parameters = pipe.transformer.parameters()
    all_params = torch.nn.utils.parameters_to_vector(model_parameters)
    with torch.no_grad():
        all_params_cpu = all_params.abs().to('cpu').sort().values
        threshold_idx = int(rate * len(all_params_cpu))
        threshold = all_params_cpu[threshold_idx]
    with torch.no_grad():
        for param in pipe.transformer.parameters():
            mask = torch.abs(param) < threshold
            param.data[mask] = 0.0
    modified_transformer_state_dict = pipe.transformer.state_dict()
    os.makedirs(pruning_weight_path, exist_ok=True)
    save_path = os.path.join(pruning_weight_path, f"transformer_pruning_{rate}.pth")
    torch.save(modified_transformer_state_dict, save_path)

def save_image(img, suffix, save_path):
    if isinstance(img, torch.Tensor) and img.dtype == torch.bfloat16:
        img = img.float()
    img_grid = image_grid(img)
    img_grid.save(os.path.join(save_path, f'{suffix}.png'))

def create_single_row_image(image_paths, gap=15, bg_color=(0, 0, 0)):
    images = []
    for p in image_paths:
        try:
            images.append(Image.open(p))
        except Exception as e:
            print(f"Warning: skipping file {p}: {e}")
    if not images:
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
    x0 = load_512(image_path, *offsets, device).to(device, dtype=default_dtype)
    ldm = load_model_pixart(model_id, device, torch_dtype=default_dtype)
    # scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    scheduler = DDIMScheduler.from_config("./pretrained_models/stable-diffusion-v1-4", subfolder="scheduler")
    ldm.scheduler = scheduler
    ldm.scheduler.set_timesteps(num_diffusion_steps)

    # with autocast(device):
    w0 = (ldm.vae.encode(x0).latent_dist.mode() * 0.18215).float().to(device, dtype=default_dtype)

    # Inversion
    _, zs, wts = fingerInv_ldm_pixart(
        ldm, w0, etas=1.0, prompt=prompt_src, cfg_scale=cfg_src,
        prog_bar=True, num_inference_steps=num_diffusion_steps
    )
    zs_tensor = zs

    # ==== Original ====
    # with autocast(device):
    w0_original, _ = verification_generating_ldm_pixart(
        ldm, xT=wts[num_diffusion_steps], etas=1.0,
        prompts=[prompt_tar], cfg_scales=[cfg_tar],
        prog_bar=True, zs=zs_tensor[:num_diffusion_steps], controller=None
    )
    x0_original = ldm.vae.decode(1 / 0.18215 * w0_original).sample
    save_image(x0_original, "original", save_path)

    # ==== Pruning ====
    for pruning_rate in pruning_rates:
        pruned_weight_file = os.path.join(pruning_weight_path, f"transformer_pruning_{pruning_rate}.pth")
        if not os.path.exists(pruned_weight_file):
            prune_transformer_for_pixart(ldm, pruning_rate, pruning_weight_path)
        ldm.transformer.load_state_dict(torch.load(pruned_weight_file, map_location=device))
        # with autocast(device):
        w0_pruned, _ = verification_generating_ldm_pixart(
            ldm, xT=wts[num_diffusion_steps], etas=1.0,
            prompts=[prompt_tar], cfg_scales=[cfg_tar],
            prog_bar=True, zs=zs_tensor[:num_diffusion_steps], controller=None
        )
        x0_pruned = ldm.vae.decode(1 / 0.18215 * w0_pruned).sample
        save_image(x0_pruned.float(), f"pruned_{pruning_rate}", save_path)

    # ==== Quantization ====
    for precision in quantized_precisions:
        ldm_quant = load_model_pixart(model_id, device, torch_dtype=precision)
        ldm_quant.scheduler = scheduler
        ldm_quant.scheduler.set_timesteps(num_diffusion_steps)
        # q_wts = [w.to(device, dtype=precision) for w in wts]
        # q_zs_tensor = torch.stack([z.to(device, dtype=precision) for z in zs])
        q_wts = wts
        q_zs_tensor = zs
        # with autocast(device_type="cuda", dtype=precision):
        with autocast(dtype=default_dtype), inference_mode():
            w0_quant, _ = verification_generating_ldm_pixart(
                ldm_quant,
                xT=q_wts[num_diffusion_steps],
                etas=1.0,
                prompts=[prompt_tar],
                cfg_scales=[cfg_tar],
                prog_bar=True,
                zs=q_zs_tensor[:num_diffusion_steps],
                controller=None,
                torch_dtype=precision
            )
            x0_quant = ldm_quant.vae.decode(1 / 0.18215 * w0_quant).sample
        save_image(x0_quant.float(), f"quantized_{precision}", save_path)

    # ==== Combine images ====
    suffix_list = ["original"] + \
                  [f"pruned_{rate}" for rate in pruning_rates] + \
                  [f"quantized_{p}" for p in quantized_precisions]
    all_image_paths = [os.path.join(save_path, f) + ".png" for f in suffix_list if os.path.exists(os.path.join(save_path, f + ".png"))]
    combined_img = create_single_row_image(all_image_paths, gap=15, bg_color=(0, 0, 0))
    if combined_img:
        combined_img.save(os.path.join(save_path, 'combined_image_row.png'))
        print(f"All results are saved into: {os.path.join(save_path, 'combined_image_row.png')}")
    else:
        print("No images generated for combination.")
