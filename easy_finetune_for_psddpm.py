from dataclasses import dataclass
from datasets import load_dataset
from torchvision import transforms
import torch
import os, glob
from torch import autocast, inference_mode
from diffusers import DDPMPipeline, DDIMScheduler
from tqdm import tqdm
import torch.nn.functional as F

model_path = './pretrained_models/ps_ddpm/'
model_ids = [
    "ddpm-ema-bedroom-256",
    "ddpm-ema-cat-256",
    "ddpm-ema-celebahq-256",
    "ddpm-ema-church-256"
]
device = "cuda:0"

dataset_name = "./images/finetune_ds_laion"
png_files = glob.glob(dataset_name + '/train/*.png')
dataset = load_dataset('imagefolder', data_files=png_files, split='train')

image_size = 256
batch_size = 1
preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)

train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)

for model_id in model_ids:
    image_pipe = DDPMPipeline.from_pretrained(model_path + model_id)
    image_pipe.to(device)

    num_epochs = 1
    lr = 1e-6
    grad_accumulation_steps = 1
    optimizer = torch.optim.AdamW(image_pipe.unet.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            clean_images = batch["images"].to(device)

            # Add random noise to images (forward diffusion)
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0,
                image_pipe.scheduler.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()
            noisy_images = image_pipe.scheduler.add_noise(clean_images, noise, timesteps)

            # Predict noise residual
            noise_pred = image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]

            # Calculate MSE loss between predicted and true noise
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            # Optimize model parameters
            if (step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Periodically save model and print loss
            if (step + 1) % 100 == 0:
                image_pipe.save_pretrained(f"./attacked_weights/psddpm/finetuning/{model_id}_ft_step_{step+1}")
                print(f"Step {step+1} loss: {loss.item()}")

            if (step + 1) % 1000 == 0:
                break

    # Optional: Save final model if needed
    # image_pipe.save_pretrained("my-finetuned-model-epoch" + str(num_epochs))
