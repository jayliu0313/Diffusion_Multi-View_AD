import torch
import torch.nn as nn
import argparse
import os
import os.path as osp
import math
import itertools
from tqdm import tqdm

from core.models.adapter import Adapter
from core.data import train_lightings_loader, val_lightings_loader

import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import CLIPProcessor, CLIPVisionModel
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
# from diffusers import StableDiffusionImg2ImgPipeline

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--data_path', default="/mnt/home_6T/public/jayliu0313/datasets/Eyecandies/", type=str)
parser.add_argument('--ckpt_path', default="checkpoints/rgb_checkpoints/finetune_condition_diffusion_unet4")
# parser.add_argument('--diffusion_id', default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--load_unet_ckpt", default=None)
parser.add_argument("--load_adapter_ckpt", default=None)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--val_every', default=3, type=int)

# Training Setup
parser.add_argument("--rand_weight", default=0.3, type=float)
parser.add_argument("--training_mode", default="fuse_fc", help="traing type: mean, fuse_fc, fuse_both, random_both")
parser.add_argument("--model", default="Masked_Conv", help="traing type: Conv, Conv_Ins, Masked_Conv")
parser.add_argument("--learning_rate", default=5e-6)
parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
parser.add_argument("--workers", default=8)
parser.add_argument('--CUDA', type=int, default=0, help="choose the device of CUDA")
parser.add_argument("--lr_scheduler", type=str, default="constant", help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'' "constant", "constant_with_warmup"]'),)
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument('--epoch', default=0, type=int, help="Which epoch to start training at")
parser.add_argument("--num_train_epochs", type=int, default=1000)
parser.add_argument(
    "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
)
parser.add_argument(
    "--max_train_steps",
    type=int,
    default=None,
    help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
)
parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--gradient_checkpointing",
    action="store_true",
    help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
)

class Pipline():
    def __init__(self, weight_dtype=torch.float32, clip_id="openai/clip-vit-base-patch32", diffusion_id="CompVis/stable-diffusion-v1-4", revision="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c"):
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = weight_dtype
        # self.weight_type = weight_type
        # Load CLIP Image Encoder
        self.clip_encoder = CLIPVisionModel.from_pretrained(clip_id).to(self.device, dtype=weight_dtype)
        self.clip_encoder.requires_grad_(False)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_id)

        self.adapter = Adapter().to(self.device)
        # Load models and create wrapper for stable diffusion
        self.vae = AutoencoderKL.from_pretrained(
                    diffusion_id,
                    subfolder="vae",
                    revision=revision,
                    torch_dtype=weight_dtype
                ).to(self.device)
        self.vae.requires_grad_(False)

        self.unet = UNet2DConditionModel.from_pretrained(
            diffusion_id,
            subfolder="unet",
            revision=revision,
            # torch_dtype=weight_dtype
        ).to(self.device)
        
        self.noise_scheduler = DDPMScheduler.from_config(diffusion_id, subfolder="scheduler")
        self.load_ckpt(args.load_unet_ckpt, args.load_adapter_ckpt)
        print("current device:", self.device)
        
    def load_ckpt(self, unet_ckpt, adapter_ckpt):
        if unet_ckpt is not None:
            self.unet.load_state_dict(torch.load(unet_ckpt, map_location=self.device))
            print("load unet ckpt from", unet_ckpt)
        if adapter_ckpt is not None:
            self.adapter.load_state_dict(torch.load(adapter_ckpt, map_location=self.device))
            print("load adapter ckpt from", adapter_ckpt)
             
    def get_vae_latents(self, x):
        x = x.float() * 2.0 - 1.0
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * 0.18215
        return latents
    
    def get_clip_embedding(self, x):
        inputs = self.clip_processor(images=x, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        clip_hidden_states = self.clip_encoder(**inputs).last_hidden_state
        return clip_hidden_states
    
    def forward_process(self, latents):
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device, dtype=self.dtype)
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        return noise, timesteps, noisy_latents
    
    def run(self, args):
        train_log_file = open(osp.join(args.ckpt_path, "training_log.txt"), "a", 1)
        val_log_file = open(osp.join(args.ckpt_path, "val_log.txt"), "a", 1)

        train_dataloader = train_lightings_loader(args)
        val_dataloader = val_lightings_loader(args)
    
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True
            
        params_to_optimize = (
            itertools.chain(self.unet.parameters(), self.adapter.parameters(),)
        )
            
        optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-08)    
        
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=len(train_dataloader) * args.num_train_epochs,
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        print("num_train_epochs:", args.num_train_epochs)

        best_val_loss = float('inf')
        for epoch in range(args.epoch, args.num_train_epochs):
            self.unet.train()
            self.adapter.train()
            epoch_loss = 0.0

            for lightings, _ in tqdm(train_dataloader):
                lightings = lightings.view(-1, 3, args.image_size, args.image_size)

                embedding = self.get_clip_embedding(lightings)
                # Get CLIP embeddings
                embedding = self.adapter(embedding) 
                
                latents = self.get_vae_latents(lightings.to(self.device, dtype=self.dtype))
                noise, timesteps, noisy_latents = self.forward_process(latents)

                # Diffusion process
                model_pred = self.unet(noisy_latents, timesteps, embedding).sample
                target = noise
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                epoch_loss += loss.item()
                    
                loss.backward()
                nn.utils.clip_grad_norm_(self.unet.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                
                
            epoch_loss /= len(train_dataloader)
            print('Training - Epoch {}: Loss: {:.6f}'.format(epoch, epoch_loss))
            train_log_file.write('Training - Epoch {}: Loss: {:.6f}\n'.format(epoch, epoch_loss))
            
            # evaluation
            if epoch % args.val_every == 0 or epoch == args.num_train_epochs - 1:
                self.unet.eval()
                self.adapter.eval()
                epoch_val_loss = 0.0

                for lightings, _ in tqdm(val_dataloader):
                    with torch.no_grad():
                        lightings = lightings.reshape(-1, 3, args.image_size, args.image_size).to(self.device)
                
                        latents = self.get_vae_latents(lightings)
                        noise, timesteps, noisy_latents = self.forward_process(latents)
                        embedding = self.get_clip_embedding(lightings)
                        
                        # Get CLIP embeddings
                        embedding = self.adapter(embedding)   
                        
                        model_pred = self.unet(noisy_latents, timesteps, embedding).sample
                        target = noise
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        
                        epoch_val_loss += loss.item()
                        
                epoch_val_loss /= len(val_dataloader)            
                print('Validation - Epoch {}: Loss: {:.6f}'.format(epoch, epoch_val_loss,))
                val_log_file.write('Validation - Epoch {}: Loss: {:.6f}\n'.format(epoch, epoch_val_loss))
        
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    model_path = args.ckpt_path+f'/best_unet_ckpt.pth'
                    torch.save(self.unet.state_dict(), model_path)
                    model_path = args.ckpt_path+f'/best_adapter_ckpt.pth'
                    torch.save(self.adapter.state_dict(), model_path)
                    print("Save the best checkpoint")
        
        val_log_file.close()
        train_log_file.close()

    
if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
        
    Training = Pipline()
    Training.run(args)