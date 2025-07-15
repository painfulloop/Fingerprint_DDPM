# Code for Fingerprinting Denoising Diffusion Probabilistic Models

This repository contains code for the CVPR 2025 paper ["Fingerprinting Denoising Diffusion Probabilistic Models"](https://openaccess.thecvf.com/content/CVPR2025/html/Teng_Fingerprinting_Denoising_Diffusion_Probabilistic_Models_CVPR_2025_paper.html). 

> **Note**: Due to original device unavailability, this project has been reproduced and simplified on new hardware (H20 GPU, 96G VRAM).

The code development refers to DDPM inversion. Thanks to them: https://github.com/inbarhub/DDPM_inversion.

The download links for different pretrained models as well as the uniqueness and robustness analysis are provided in eval.sh.

## Pretrained Models & Evaluation

### Pixel-Space DDPMs (PS-DDPMs)

Download weights to `./pretrained_models/ps_ddpms/`:
- cat: https://huggingface.co/google/ddpm-ema-cat-256
- church: https://huggingface.co/google/ddpm-ema-church-256
- bedroom: https://huggingface.co/google/ddpm-ema-bedroom-256
- celebahq: https://huggingface.co/google/ddpm-ema-celebahq-256

#### Uniqueness analysis:

```python ps_ddpm_uniqueness.py```

#### Robustness analysis:

Run `python easy_finetune_for_psddpm.py` to simply finetune and disturb these pixel-space DDPMs. You can choose your own small dataset or use any subset of the laion dataset. 

Then, run robustness evaluations:

```python ps_ddpm_robustness.py```

### Latent Diffusion Models (LDMs)

Download weights to `./pretrained_models/`:
- SD v1.4: https://huggingface.co/CompVis/stable-diffusion-v1-4
- Pixart: https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512
- Deci: https://huggingface.co/Deci/DeciDiffusion-v1-0

#### Uniqueness analysis:

```python ldm_uniqueness.py```

#### Robustness analysis:
Download weights to `./attack_weights/ldm/finetuning/` for finetuning analysis of SD:
- https://huggingface.co/xyn-ai/anything-v4.0
- https://huggingface.co/XpucT/Deliberate
- https://huggingface.co/SG161222/Realistic_Vision_V2.0 (Realistic_Vision_V2.0.ckpt)
- https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5

Then, run robustness evaluations:

```
python ldm_robustness_SD.py

python ldm_robustness_Deci.py

python ldm_robustness_Pixart.py
```


If you find the code useful and use this code for your research, please cite our paper:
 ```
@InProceedings{Teng_2025_CVPR,
author = {Teng, Huan and Quan, Yuhui and Wang, Chengyu and Huang, Jun and Ji, Hui},
title = {Fingerprinting Denoising Diffusion Probabilistic Models},
booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
month = {June},
year = {2025},
pages = {28811-28820}
}
 ```
