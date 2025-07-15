import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, PixArtAlphaPipeline
from fingerprint_inversion.fingerinv_utils import (
    fingerInv_ldm,
    verification_generating_ldm,
    fingerInv_ldm_pixart,
    verification_generating_ldm_pixart,
)
from fingerprint_inversion.utils import image_grid, load_512
from torch import autocast, inference_mode
from PIL import Image

# ==== Config ====
device = "cuda:0"
model_paths = {
    "stable": "./pretrained_models/stable-diffusion-v1-4",
    "pixart": "./pretrained_models/pixart",
    "deci": "./pretrained_models/DeciDiffusion-v1-0"
}
qr_len = '64'
image_path = './images/qr_code_random_' + qr_len + '.png'
save_path = './results/ldm_uniqueness_' + qr_len
prompt_tar = prompt_src = ""
cfg_tar = cfg_src = 0.0
num_diffusion_steps = 20
os.makedirs(save_path, exist_ok=True)
model_ids = ["stable", "pixart", "deci"]

# ==== Utils ====
def load_model(model_id, device, pipe_class=StableDiffusionPipeline, torch_dtype=None, custom_pipeline=None):
    kwargs = {}
    if torch_dtype is not None:
        kwargs['torch_dtype'] = torch_dtype
    if custom_pipeline is not None:
        kwargs['custom_pipeline'] = custom_pipeline
    model = pipe_class.from_pretrained(model_id, **kwargs).to(device)
    return model
def save_image(img, suffix, save_path):
    img_grid = image_grid(img)
    img_grid.save(os.path.join(save_path, f'{suffix}.png'))
def create_combined_image(save_path, model_ids, img_size=512, spacing=10):
    size = (3 * img_size + 2 * spacing, 3 * img_size + 2 * spacing)
    big_image = Image.new('RGB', size)
    for i, src in enumerate(model_ids):
        for j, tar in enumerate(model_ids):
            fn = f"{src}_to_{tar}.png"
            img_path = os.path.join(save_path, fn)
            if os.path.exists(img_path):
                img = Image.open(img_path)
                x, y = j*(img_size+spacing), i*(img_size+spacing)
                big_image.paste(img, (x, y))
    big_image.save(os.path.join(save_path, "combined_image.png"))

# ==== Main ====
if __name__ == "__main__":
    offsets = (0, 0, 0, 0)
    x0 = load_512(image_path, *offsets, device)

    # load model
    ldm_stable = load_model(model_paths["stable"], device)
    ldm_pixart = load_model(model_paths["pixart"], device, pipe_class=PixArtAlphaPipeline, torch_dtype=torch.float32)
    ldm_deci = load_model(model_paths["deci"], device, torch_dtype=torch.float16, custom_pipeline=model_paths["deci"])
    ldm_deci.unet = ldm_deci.unet.from_pretrained(
        model_paths["deci"], subfolder='flexible_unet', torch_dtype=torch.float16
    ).to(device)

    scheduler = DDIMScheduler.from_config(model_paths["stable"], subfolder="scheduler")
    for m in [ldm_stable, ldm_deci, ldm_pixart]:
        m.scheduler = scheduler
        m.scheduler.set_timesteps(num_diffusion_steps)

    # fingerinv
    with autocast("cuda"), inference_mode():
        w0_deci = (ldm_deci.vae.encode(x0).latent_dist.mode() * 0.18215).float()
        w0_stable = (ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215)
        w0_pixart = (ldm_pixart.vae.encode(x0).latent_dist.mode() * 0.18215).float()

    _, zs_pixart, wts_pixart = fingerInv_ldm_pixart(
        ldm_pixart, w0_pixart, etas=1.0, prompt=prompt_src, cfg_scale=cfg_src,
        prog_bar=True, num_inference_steps=num_diffusion_steps)
    _, zs_deci, wts_deci = fingerInv_ldm(
        ldm_deci, w0_deci, etas=1.0, prompt=prompt_src, cfg_scale=cfg_src,
        prog_bar=True, num_inference_steps=num_diffusion_steps, dtype=torch.float16)
    _, zs_stable, wts_stable = fingerInv_ldm(
        ldm_stable, w0_stable, etas=1.0, prompt=prompt_src, cfg_scale=cfg_src,
        prog_bar=True, num_inference_steps=num_diffusion_steps)


    # cross-generating
    w0_stable_rev_dc, _ = verification_generating_ldm(
        ldm_stable, xT=wts_deci[num_diffusion_steps], etas=1.0, prompts=[prompt_tar],
        cfg_scales=[cfg_tar], prog_bar=True, zs=zs_deci[:num_diffusion_steps], controller=None, dtype=torch.float16)
    save_image(ldm_stable.vae.decode(1 / 0.18215 * w0_stable_rev_dc).sample, "deci_to_stable", save_path)

    w0_pixart_rev_deci, _ = verification_generating_ldm_pixart(
        ldm_pixart, xT=wts_deci[num_diffusion_steps], etas=1.0, prompts=[prompt_tar],
        cfg_scales=[cfg_tar], prog_bar=True, zs=zs_deci[:num_diffusion_steps], controller=None, torch_dtype=torch.float16)
    save_image(ldm_pixart.vae.decode(1 / 0.18215 * w0_pixart_rev_deci).sample, "deci_to_pixart", save_path)

    w0_deci_rev_deci, _ = verification_generating_ldm(
        ldm_deci, xT=wts_deci[num_diffusion_steps], etas=1.0, prompts=[prompt_tar],
        cfg_scales=[cfg_tar], prog_bar=True, zs=zs_deci[:num_diffusion_steps], controller=None, dtype=torch.float16)
    save_image(ldm_deci.vae.decode(1 / 0.18215 * w0_deci_rev_deci).sample, "deci_to_deci", save_path)

    w0_stable_rev_sd, _ = verification_generating_ldm(
        ldm_stable, xT=wts_stable[num_diffusion_steps], etas=1.0, prompts=[prompt_tar],
        cfg_scales=[cfg_tar], prog_bar=True, zs=zs_stable[:num_diffusion_steps], controller=None)
    save_image(ldm_stable.vae.decode(1 / 0.18215 * w0_stable_rev_sd).sample, "stable_to_stable", save_path)

    w0_deci_rev_sd, _ = verification_generating_ldm(
        ldm_deci, xT=wts_stable[num_diffusion_steps], etas=1.0, prompts=[prompt_tar],
        cfg_scales=[cfg_tar], prog_bar=True, zs=zs_stable[:num_diffusion_steps], controller=None, dtype=torch.float16)
    save_image(ldm_deci.vae.decode(1 / 0.18215 * w0_deci_rev_sd).sample, "stable_to_deci", save_path)

    w0_pixart_rev_sd, _ = verification_generating_ldm_pixart(
        ldm_pixart, xT=wts_stable[num_diffusion_steps], etas=1.0, prompts=[prompt_tar],
        cfg_scales=[cfg_tar], prog_bar=True, zs=zs_stable[:num_diffusion_steps], controller=None)
    save_image(ldm_pixart.vae.decode(1 / 0.18215 * w0_pixart_rev_sd).sample, "stable_to_pixart", save_path)

    w0_stable_rev_pixart, _ = verification_generating_ldm(
        ldm_stable, xT=wts_pixart[num_diffusion_steps], etas=1.0, prompts=[prompt_tar],
        cfg_scales=[cfg_tar], prog_bar=True, zs=zs_pixart[:num_diffusion_steps], controller=None)
    save_image(ldm_stable.vae.decode(1 / 0.18215 * w0_stable_rev_pixart).sample, "pixart_to_stable", save_path)

    w0_pixart_rev_pixart, _ = verification_generating_ldm_pixart(
        ldm_pixart, xT=wts_pixart[num_diffusion_steps], etas=1.0, prompts=[prompt_tar],
        cfg_scales=[cfg_tar], prog_bar=True, zs=zs_pixart[:num_diffusion_steps], controller=None)
    save_image(ldm_pixart.vae.decode(1 / 0.18215 * w0_pixart_rev_pixart).sample, "pixart_to_pixart", save_path)

    w0_deci_rev_pixart, _ = verification_generating_ldm(
        ldm_deci, xT=wts_pixart[num_diffusion_steps], etas=1.0, prompts=[prompt_tar],
        cfg_scales=[cfg_tar], prog_bar=True, zs=zs_pixart[:num_diffusion_steps], controller=None, dtype=torch.float16)
    save_image(ldm_deci.vae.decode(1 / 0.18215 * w0_deci_rev_pixart).sample, "pixart_to_deci", save_path)

    create_combined_image(save_path, model_ids, img_size=512, spacing=10)
