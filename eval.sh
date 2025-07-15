# For pixel-space DDPMs:
# download weights to ./pretrained_models/ps_ddpms/
# cat: https://huggingface.co/google/ddpm-ema-cat-256/tree/main
# church: https://huggingface.co/google/ddpm-ema-church-256/tree/main
# bedroom: https://huggingface.co/google/ddpm-ema-bedroom-256/tree/main
# celebahq: https://huggingface.co/google/ddpm-ema-celebahq-256/tree/main

# Uniqness analysis:
python ps_ddpm_uniqueness.py
# Robustness analysis:
# run easy_finetune_for_psddpm.py to disturb ps_ddpms, you can choose your own small dataset or use any subset of laion dataset.
# python easy_finetune_for_psddpm.py
python ps_ddpm_robustness.py

# For LDMs:
# download weights to ./pretrained_models/
# SD v1.4: https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main
# Pixart: https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512/tree/main
# Deci: https://huggingface.co/Deci/DeciDiffusion-v1-0/tree/main

# Uniqness analysis:
python ldm_uniqueness.py

# Robustness analysis:
# Download weights to ./attack_weights/ldm/finetuning/ for finetuning analysis of SD:
# https://huggingface.co/xyn-ai/anything-v4.0/tree/main
# https://huggingface.co/XpucT/Deliberate
# https://huggingface.co/SG161222/Realistic_Vision_V2.0/tree/main  #(Realistic_Vision_V2.0.ckpt)
# https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main
python ldm_robustness_SD.py
python ldm_robustness_Deci.py
python ldm_robustness_Pixart.py