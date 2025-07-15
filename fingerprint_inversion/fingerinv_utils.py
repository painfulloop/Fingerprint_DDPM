import torch, random
import os, numpy as np
from tqdm import tqdm
from diffusers import PixArtAlphaPipeline
import torch.nn as nn
from torch.cuda.amp import autocast


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )
    return latent_image_ids.to(device=device, dtype=dtype)

def load_real_image(folder = "data/", img_name = None, idx = 0, img_size=512, device='cuda'):
    from ddm_inversion.utils import pil_to_tensor
    from PIL import Image
    from glob import glob
    if img_name is not None:
        path = os.path.join(folder, img_name)
    else:
        path = glob(folder + "*")[idx]

    img = Image.open(path).resize((img_size,
                                    img_size))

    img = pil_to_tensor(img).to(device)

    if img.shape[1]== 4:
        img = img[:,:3,:,:]
    return img

def mu_tilde(model, xt,x0, timestep):
    "mu_tilde(x_t, x_0) DDPM paper eq. 7"
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_t = model.scheduler.alphas[timestep]
    beta_t = 1 - alpha_t 
    alpha_bar = model.scheduler.alphas_cumprod[timestep]
    return ((alpha_prod_t_prev ** 0.5 * beta_t) / (1-alpha_bar)) * x0 +  ((alpha_t**0.5 *(1-alpha_prod_t_prev)) / (1- alpha_bar))*xt

def tv_loss(x):
    pixel_dif1 = x[:, :, 1:, :] - x[:, :, :-1, :]
    pixel_dif2 = x[:, :, :, 1:] - x[:, :, :, :-1]
    return torch.sum(torch.abs(pixel_dif1)) + torch.sum(torch.abs(pixel_dif2))

def get_noise(x0, model, noise_ini, alphas, alpha_bar, sqrt_one_minus_alpha_bar, t, uncond_embedding, idx=13, num_inference_steps=20, model_type='LDM'):
    device = model.device
    dtype = model.dtype

    # noise initialization
    noise =  20*((idx-1)/num_inference_steps) *2*(torch.rand_like(x0)-0.5) + torch.randn_like(x0)

    noise = noise.to(device).to(dtype)
    noise.requires_grad = True
    alphas = alphas.to(device).to(dtype)
    alpha_bar = alpha_bar.to(device).to(dtype)
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.to(device).to(dtype)

    for param in model.unet.parameters():
        param.requires_grad = True
    
    with autocast():

        for i in tqdm(range(10)):
            xt = (x0.to(device) * (alpha_bar[int(t)] ** 0.5) - noise * sqrt_one_minus_alpha_bar[int(t)]).to(device).to(dtype)
            
            # calculating model output
            if model_type == 'LDM':
                out = model.unet.forward(xt, timestep=t.to(dtype), encoder_hidden_states=uncond_embedding.to(dtype))
                noise_pred = out.sample
            elif model_type == 'DDPM':
                t = torch.tensor(t, dtype=torch.float32)
                out = model.unet.forward(xt, timestep=t.to(dtype))
                noise_pred = out.sample
            # optimizing noise
            loss = (1-(idx-1)/num_inference_steps)*nn.MSELoss(reduction='mean')(noise_pred, noise) - 1*((idx-1)/num_inference_steps) * tv_loss(xt)
            # print(loss.item())
            loss.backward(retain_graph=True)
            with torch.no_grad():
                noise -= 0.1 * noise.grad
            noise.grad.zero_()

    return noise.to(dtype)

def get_noise_pixart(x0, model, noise_ini, alphas, alpha_bar, sqrt_one_minus_alpha_bar, t, uncond_embedding, idx=13, num_inference_steps=20):
    device = model.device
    dtype = model.dtype
    for param in model.transformer.parameters():
        param.requires_grad = True

    # noise initialization
    noise =  20*((idx-1)/num_inference_steps) *2*(torch.rand_like(x0)-0.5) + torch.randn_like(x0)
    noise = noise.to(device).to(dtype)
    noise.requires_grad = True

    if not torch.is_tensor(t):
        is_mps = x0.device.type == "mps"
        t_tensor = torch.tensor([t], dtype=torch_dtype if is_mps else torch.float64, device=x0.device)
    else:
        t_tensor = t
    
    t_tensor = t_tensor.expand(x0.shape[0]).to(model.device, dtype=dtype)

    for i in tqdm(range(10)):
        xt = (x0 * (alpha_bar[t] ** 0.5) - noise * sqrt_one_minus_alpha_bar[t]).to(device).to(dtype)
        
        # calculating model output
        out = model.transformer.forward(xt, timestep=t_tensor.to(dtype), added_cond_kwargs={"resolution": None, "aspect_ratio": None},encoder_hidden_states=uncond_embedding.to(dtype), return_dict=False)[0]
        noise_pred = out
        if model.transformer.config.out_channels // 2 == model.transformer.config.in_channels:
            noise_pred = noise_pred.chunk(2, dim=1)[0]

        # optimizing noise
        loss = (1-(idx-1)/num_inference_steps)*nn.MSELoss(reduction='mean')(noise_pred, noise) - 1*((idx-1)/num_inference_steps)*tv_loss(xt)
        # print(loss.item())
        loss.backward(retain_graph=True)
        with torch.no_grad():
            noise -= 0.1 * noise.grad
        noise.grad.zero_()

    return noise
def sample_xts_from_x0(model, x0, num_inference_steps=20,uncond_embedding=None,model_type='LDM'):
    """
    Samples from P(x_1:T|x_0)
    """
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
    alphas = model.scheduler.alphas
    betas = 1 - alphas
    sqrt_one_minus_betas = (1-betas) ** 0.5
    variance_noise_shape = (
            num_inference_steps,
            model.unet.in_channels, 
            model.unet.sample_size,
            model.unet.sample_size)
    
    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
    xts = torch.zeros((num_inference_steps+1,model.unet.in_channels, model.unet.sample_size, model.unet.sample_size)).to(x0.device)
    xts[0] = x0

    for t in reversed(timesteps):
        idx = num_inference_steps-t_to_idx[int(t)]
        noise_t = get_noise(x0, model, noise_ini=None, alphas = alphas, alpha_bar=alpha_bar, sqrt_one_minus_alpha_bar=sqrt_one_minus_alpha_bar, t=t, \
            uncond_embedding=uncond_embedding, idx=idx, num_inference_steps=num_inference_steps, model_type=model_type)

        xts[idx] = x0 * (alpha_bar[t] ** 0.5)  - noise_t * sqrt_one_minus_alpha_bar[t]

    return xts
def sample_xts_from_x0_pixart(model, x0, num_inference_steps=50,uncond_embedding=None):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
    alphas = model.scheduler.alphas
    betas = 1 - alphas
    sqrt_one_minus_betas = (1-betas) ** 0.5
    variance_noise_shape = (
            num_inference_steps,
            model.transformer.config.in_channels, 
            model.transformer.config.sample_size,
            model.transformer.config.sample_size)

    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
    xts = torch.zeros((num_inference_steps+1,model.transformer.config.in_channels, model.transformer.config.sample_size, model.transformer.config.sample_size)).to(x0.device)
    xts[0] = x0

    for t in reversed(timesteps):
        idx = num_inference_steps-t_to_idx[int(t)]
 
        noise_t = get_noise_pixart(x0, model, noise_ini=None, alphas = alphas, alpha_bar=alpha_bar, sqrt_one_minus_alpha_bar=sqrt_one_minus_alpha_bar, t=t, \
            uncond_embedding=uncond_embedding, idx=idx, num_inference_steps=num_inference_steps)
        xts[idx] = x0 * (alpha_bar[t] ** 0.5)  - noise_t * sqrt_one_minus_alpha_bar[t]      
    return xts
def encode_text(model, prompts):
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length, 
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_encoding

def forward_step(model, model_output, timestep, sample):
    next_timestep = min(model.scheduler.config.num_train_timesteps - 2,
                        timestep + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps)

    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    next_sample = model.scheduler.add_noise(pred_original_sample,
                                    model_output,
                                    torch.LongTensor([next_timestep]))
    return next_sample


def get_variance(model, timestep): 
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance

def fingerInv_ldm_pixart(model, x0, 
                            etas=None,    
                            prog_bar=False,
                            prompt="",
                            cfg_scale=3.5,
                            num_inference_steps=50, 
                            eps=None,
                            dtype=torch.float32): 
    device = model.device
    if not prompt == "":
        text_embeddings = encode_text(model, prompt).to(dtype)  
    uncond_embedding = encode_text(model, "").to(dtype) 
    timesteps = model.scheduler.timesteps.to(device)

    variance_noise_shape = (
        num_inference_steps,
        model.transformer.config.in_channels, 
        model.transformer.config.sample_size,
        model.transformer.config.sample_size
    )

    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]: 
            etas = [etas] * model.scheduler.num_inference_steps
        
        etas = torch.tensor(etas, dtype=dtype, device=device) 
        xts = sample_xts_from_x0_pixart(model, x0, num_inference_steps=num_inference_steps,uncond_embedding=uncond_embedding).to(dtype) 
        alpha_bar = model.scheduler.alphas_cumprod.to(dtype)  
        zs = torch.zeros(size=variance_noise_shape, device=device, dtype=dtype)  

    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0.to(dtype).to(device) 

    op = tqdm(timesteps) if prog_bar else timesteps

    for t in op:
        idx = num_inference_steps - t_to_idx[int(t)] - 1
        if not eta_is_zero:
            xt = xts[idx + 1][None].to(dtype).to(device) 

        with torch.no_grad():
            t_tensor = t.expand(xt.shape[0]).to(device)  
            
            out = model.transformer.forward(xt.to(dtype), timestep=t_tensor.to(dtype), 
                added_cond_kwargs={"resolution": None, "aspect_ratio": None}, 
                encoder_hidden_states=uncond_embedding.to(dtype), return_dict=False)[0]
            
            if not prompt == "":
                cond_out = model.transformer.forward(xt.to(dtype), timestep=t_tensor.to(dtype), 
                    added_cond_kwargs={"resolution": None, "aspect_ratio": None}, 
                    encoder_hidden_states=text_embeddings.to(dtype), return_dict=False)[0]

        if not prompt == "":
            noise_pred = out + cfg_scale * (cond_out - out)
        else:
            noise_pred = out

        # learned sigma
        if model.transformer.config.out_channels // 2 == model.transformer.config.in_channels:
            noise_pred = noise_pred.chunk(2, dim=1)[0]

        if eta_is_zero:
            xt = forward_step(model, noise_pred.to(dtype), t, xt).to(dtype)  
        else:
            xtm1 = xts[idx][None].to(dtype).to(device) 
            
            # pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5
            pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / (alpha_bar[t] ** 0.5 + 1e-5)
            
            prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
            alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep].to(dtype) if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod.to(dtype)

            variance = get_variance(model, t).to(dtype) 
            # pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance) ** 0.5 * noise_pred
            pred_sample_direction = torch.relu(1 - alpha_prod_t_prev - etas[idx] * variance) ** 0.5 * noise_pred
            mu_xt = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
            
            z = (xtm1 - mu_xt) / (etas[idx] * variance ** 0.5 + 1e-5)

            if torch.isnan(z).any() or torch.isnan(mu_xt).any() or torch.isnan(pred_original_sample).any():
                print("NaN detected at timestep {t}")
            #     raise ValueError(f"NaN detected at timestep {t}")

            zs[idx] = z
            
            # correction to avoid error accumulation
            xtm1 = mu_xt + (etas[idx] * variance ** 0.5) * z
            xts[idx] = xtm1
            
    if zs is not None:
        zs[0] = torch.zeros_like(zs[0]).to(dtype)
        
    return xt, zs, xts
def fingerInv_ldm(model, x0, 
                              etas=None,    
                              prog_bar=False,
                              prompt="",
                              cfg_scale=3.5,
                              num_inference_steps=50, 
                              eps=None,
                              dtype=torch.float32):

    device = model.device
    model = model.to(dtype)

    if not prompt=="":
        text_embeddings = encode_text(model, prompt)
    uncond_embedding = encode_text(model, "")

    timesteps = model.scheduler.timesteps.to(device)
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels, 
        model.unet.sample_size,
        model.unet.sample_size)

    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
        etas = torch.tensor(etas, dtype=dtype, device=device)
        xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps,uncond_embedding=uncond_embedding).to(device).to(dtype)
        alpha_bar = model.scheduler.alphas_cumprod.to(device).to(dtype)
        zs = torch.zeros(size=variance_noise_shape, device=device, dtype=dtype)

    t_to_idx = {int(v.to(dtype)): k for k, v in enumerate(timesteps)}
    xt = x0.to(device).to(dtype)
    op = tqdm(timesteps) if prog_bar else timesteps

    for t in op:
        idx = num_inference_steps - t_to_idx[int(t)] - 1
        if not eta_is_zero:
            xt = xts[idx + 1][None].to(device).to(dtype)
                    
        with torch.no_grad():
            out = model.unet.forward(xt, timestep=t, encoder_hidden_states=uncond_embedding)
            if not prompt == "":
                cond_out = model.unet.forward(xt, timestep=t, encoder_hidden_states=text_embeddings)

        if not prompt == "":
            noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
        else:
            noise_pred = out.sample
        
        if eta_is_zero:
            xt = forward_step(model, noise_pred, t, xt).to(device).to(dtype)
        else:
            xtm1 = xts[idx][None].to(device).to(dtype)
            if alpha_bar[t] > 0:
                pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5
            else:
                pred_original_sample = x0

            prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
            alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep].to(device).to(dtype) if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod.to(device).to(dtype)
            variance = get_variance(model, t).to(device).to(dtype)
            pred_sample_direction = torch.relu(1 - alpha_prod_t_prev - etas[idx] * variance) ** 0.5 * noise_pred
            mu_xt = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
            z = (xtm1 - mu_xt) / (etas[idx] * variance ** 0.5)
            zs[idx] = z

            # correction to avoid error accumulation
            xtm1 = mu_xt + (etas[idx] * variance ** 0.5) * z
            xts[idx] = xtm1

    if not zs is None:
        zs[0] = torch.zeros_like(zs[0])

    return xt, zs, xts

def fingerInv_psddpm(model, x0, etas=None, prog_bar=False, num_inference_steps=50, eps=None, dtype=torch.float32):
    device = model.device
    
    timesteps = model.scheduler.timesteps.to(device)
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels, 
        model.unet.sample_size,
        model.unet.sample_size)

    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]: 
            etas = [etas] * model.scheduler.num_inference_steps
        etas = torch.tensor(etas, dtype=dtype, device=device)
        xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps, uncond_embedding=None, model_type='DDPM').to(device).to(dtype)
        alpha_bar = model.scheduler.alphas_cumprod.to(device).to(dtype)
        zs = torch.zeros(size=variance_noise_shape, device=device, dtype=dtype)

    t_to_idx = {int(v.to(dtype)): k for k, v in enumerate(timesteps)}
    xt = x0.to(device).to(dtype)
    op = tqdm(timesteps) if prog_bar else timesteps

    for t in op:
        idx = num_inference_steps - t_to_idx[int(t)] - 1
        if not eta_is_zero:
            xt = xts[idx + 1][None].to(device).to(dtype)
                    
        with torch.no_grad():
            out = model.unet.forward(xt, timestep=t)

        if eta_is_zero:
            xt = forward_step(model, out.sample, t, xt).to(device).to(dtype)
        else:
            xtm1 = xts[idx][None].to(device).to(dtype)
            pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * out.sample) / alpha_bar[t] ** 0.5
            prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
            alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep].to(device).to(dtype) if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod.to(device).to(dtype)
            variance = get_variance(model, t).to(device).to(dtype)
            pred_sample_direction = torch.relu(1 - alpha_prod_t_prev - etas[idx] * variance) ** 0.5 * out.sample
            mu_xt = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
            z = (xtm1 - mu_xt) / (etas[idx] * variance ** 0.5)
            zs[idx] = z
            xtm1 = mu_xt + (etas[idx] * variance ** 0.5) * z
            xts[idx] = xtm1
            
    if not zs is None:
        zs[0] = torch.zeros_like(zs[0])
        
    return xt, zs, xts

def reverse_step(model, model_output, timestep, sample, eta = 0, variance_noise=None):
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)    
    # variance = self.scheduler._get_variance(timestep, prev_timestep)
    variance = get_variance(model, timestep) #, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    # Take care of asymetric reverse process (asyrp)
    model_output_direction = model_output
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    # 8. Add noice if eta > 0
    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape, device=model.device)
        sigma_z =  eta * variance ** (0.5) * variance_noise
        prev_sample = prev_sample + sigma_z

    return prev_sample

def verification_generating_ldm(model,
                    xT, 
                    etas = 0,
                    prompts = "",
                    cfg_scales = None,
                    prog_bar = False,
                    zs = None,
                    controller=None,
                    asyrp = False,
                    dtype=torch.float32):
    model = model.to(dtype)
    batch_size = len(prompts)

    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1,1,1,1).to(model.device)

    text_embeddings = encode_text(model, prompts)
    uncond_embedding = encode_text(model, [""] * batch_size)

    if etas is None: etas = 0
    if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(batch_size, -1, -1, -1)
    op = tqdm(timesteps[-zs.shape[0]:]) if prog_bar else timesteps[-zs.shape[0]:] 

    t_to_idx = {int(v):k for k,v in enumerate(timesteps[-zs.shape[0]:])}

    for t in op:
        idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-(model.scheduler.num_inference_steps-zs.shape[0]+1)    
        ## Unconditional embedding
        with torch.no_grad():
            uncond_out = model.unet.forward(xt.to(dtype), timestep =  t, 
                                            encoder_hidden_states = uncond_embedding)

        ## Conditional embedding  
        if prompts:  
            with torch.no_grad():
                cond_out = model.unet.forward(xt.to(dtype), timestep =  t, 
                                                encoder_hidden_states = text_embeddings)
            
        
        z = zs[idx].to(dtype) if not zs is None else None
        z = z.expand(batch_size, -1, -1, -1).to(dtype)
        if prompts:
            ## classifier free guidance
            noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
        else: 
            noise_pred = uncond_out.sample
        # 2. compute less noisy image and set x_t -> x_t-1  
        xt = reverse_step(model, noise_pred, t, xt, eta = etas[idx], variance_noise = z) .to(dtype)
        if controller is not None:
            xt = controller.step_callback(xt)        
    return xt, zs
def verification_generating_psddpm(model, xT, etas=1, prog_bar=False, zs=None, controller=None, dtype=torch.float32):
    model = model.to(dtype)
    batch_size = 1  
    if etas is None: 
        etas = 0
    if type(etas) in [int, float]: 
        etas = [etas] * model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    
    timesteps = model.scheduler.timesteps.to(model.device)
    xt = xT.expand(batch_size, -1, -1, -1)
    op = tqdm(timesteps[-zs.shape[0]:]) if prog_bar else timesteps[-zs.shape[0]:] 
    t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}
    
    for t in op:
        idx = model.scheduler.num_inference_steps - t_to_idx[int(t)] - (model.scheduler.num_inference_steps - zs.shape[0] + 1)
        
        # Unconditional output
        with torch.no_grad():
            uncond_out = model.unet.forward(xt.to(dtype), timestep=t)

        z = zs[idx].to(dtype) if zs is not None else None
        z = z.expand(batch_size, -1, -1, -1).to(dtype)

        noise_pred = uncond_out.sample

        # xt -> xt-1  
        xt = reverse_step(model, noise_pred, t, xt, eta=etas[idx], variance_noise=z).to(dtype)
        
        if controller is not None:
            xt = controller.step_callback(xt)        
    
    return xt, zs

def verification_generating_ldm_pixart(model,
                    xT, 
                    etas=0,
                    prompts="",
                    cfg_scales=None,
                    prog_bar=False,
                    zs=None,
                    controller=None,
                    asyrp=False,
                    torch_dtype=torch.float32):
    model = model.to(torch_dtype)
    batch_size = len(prompts)
    
    cfg_scales_tensor = torch.Tensor(cfg_scales).to(model.device, dtype=torch_dtype).view(-1, 1, 1, 1)
    text_embeddings = encode_text(model, prompts).to(dtype=torch_dtype)
    uncond_embedding = encode_text(model, [""] * batch_size).to(dtype=torch_dtype)
    
    if etas is None:
        etas = 0
    if isinstance(etas, (int, float)):
        etas = [etas] * model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps

    timesteps = model.scheduler.timesteps.to(model.device)
    xt = xT.expand(batch_size, -1, -1, -1)
    op = tqdm(timesteps[-zs.shape[0]:]) if prog_bar else timesteps[-zs.shape[0]:]
    t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}
    
    for t in op:
        idx = model.scheduler.num_inference_steps - t_to_idx[int(t)] - (model.scheduler.num_inference_steps - zs.shape[0] + 1)
        
        if idx < 0 or idx >= len(etas):
            raise ValueError(f"Index {idx} out of bounds for etas with length {len(etas)}.")

        if not torch.is_tensor(t):
            is_mps = xt.device.type == "mps"
            t_tensor = torch.tensor([t], dtype=torch_dtype if is_mps else torch.float64, device=xt.device)
        else:
            t_tensor = t
        
        t_tensor = t_tensor.expand(xt.shape[0]).to(model.device, dtype=torch_dtype)
        
        ## Unconditional embedding
        with torch.no_grad():
            uncond_out = model.transformer.forward(
                xt, 
                timestep=t_tensor, 
                added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                encoder_hidden_states=uncond_embedding, 
                return_dict=False)[0]
        
        ## Conditional embedding  
        if prompts:  
            with torch.no_grad():
                cond_out = model.transformer.forward(
                    xt, 
                    timestep=t_tensor, 
                    added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                    encoder_hidden_states=text_embeddings, 
                    return_dict=False)[0]
        
        z = zs[idx] if zs is not None else None
        z = z.expand(batch_size, -1, -1, -1) if z is not None else z
        
        if prompts:
            ## classifier free guidance
            noise_pred = uncond_out + cfg_scales_tensor * (cond_out - uncond_out)
        else: 
            noise_pred = uncond_out
        
        # learned sigma
        if model.transformer.config.out_channels // 2 == model.transformer.config.in_channels:
            noise_pred = noise_pred.chunk(2, dim=1)[0]

        # 2. compute less noisy image and set x_t -> x_t-1  
        xt = reverse_step(model, noise_pred, int(t), xt, eta=etas[idx], variance_noise=z)
        
        if controller is not None:
            xt = controller.step_callback(xt)        
    
    return xt, zs