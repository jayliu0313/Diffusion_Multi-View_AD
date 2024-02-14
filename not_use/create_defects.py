import os
import os.path as osp
import argparse
import torch
import torch.nn as nn
import datetime
from tqdm import tqdm
import numpy as np
import random
from core.data import train_lightings_loader, val_lightings_loader
from utils.visualize_util import display_3type_image

from transformers import CLIPTextModel, AutoTokenizer
from diffusers import AutoencoderKL, UniPCMultistepScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler

import matplotlib.pyplot as plt
import matplotlib
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='visualize')
parser.add_argument('--data_path', default="/mnt/home_6T/public/jayliu0313/datasets/Eyecandies/", type=str)
parser.add_argument('--save_dir', default="visualize_defect/")
parser.add_argument("--load_unet_ckpt", default="/mnt/home_6T/public/jayliu0313/check_point/Diffusion_ckpt/unet_InputImage_TextNull_woTrainVae/best_unet.pth")
parser.add_argument('--diffusion_id', default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--revision", type=str, default="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c")
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--val_every', default=3, type=int)
parser.add_argument("--workers", default=8)

parser.add_argument("--noise_intensity", type=int, default=50)
parser.add_argument("--step_size", type=int, default=2)

class CreateDefects():
    def __init__(self, args, device):
        self.device = device
        self.bs = args.batch_size
        self.image_size = args.image_size
        self.save_dir = args.save_dir
        
        # Load training and validation data
        self.train_dataloader = train_lightings_loader(args)
        self.val_dataloader = val_lightings_loader(args)

        # Create Model
        self.tokenizer = AutoTokenizer.from_pretrained(args.diffusion_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.diffusion_id, subfolder="text_encoder")
        self.noise_scheduler = UniPCMultistepScheduler.from_pretrained(args.diffusion_id, subfolder="scheduler")

        # Load vae model
        self.vae = AutoencoderKL.from_pretrained(
                    args.diffusion_id,
                    subfolder="vae",
                    revision=args.revision,
                    torch_dtype=torch.float32
                )
            
        self.unet = UNet2DConditionModel.from_pretrained(
                args.diffusion_id,
                subfolder="unet",
                revision=args.revision)
        
        if args.load_unet_ckpt is not None:
            self.unet.load_state_dict(torch.load(args.load_unet_ckpt, map_location=self.device))
            print("Load Diffusion Model Checkpoint!!")
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.vae.to(self.device)
        self.unet.to(self.device)
        self.text_encoder.to(self.device)
        
        self.encoder_hidden_states = self.get_text_embedding("", 6*self.bs) # [6, 77, 768]
        
        self.num_inference_timesteps = int(len(self.noise_scheduler.timesteps) / args.step_size)
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)
        self.timesteps_list = self.noise_scheduler.timesteps[self.noise_scheduler.timesteps <= args.noise_intensity]
        print(self.timesteps_list)
                
    def image2latents(self, x):
        x = x.float() * 2.0 - 1.0
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * 0.18215
        return latents
    
    def latents2image(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image
    
    def forward_process_with_T(self, latents, T):
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        timesteps = torch.tensor([T], device=self.device).repeat(bsz)
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        return noise, timesteps, noisy_latents
    
    def get_text_embedding(self, text_prompt, bsz):
        tok = self.tokenizer(text_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embedding = self.text_encoder(tok.input_ids.to(self.device).repeat(bsz,1))[0]
        return text_embedding
    
    def random_crop_and_paste(self, image_tensor, texture_tensor, area_ratio = (0.02, 0.15), aspect_ratio = ((0.3, 1) , (1, 3.3)), num_pastes=5):
        batch_size, channels, height, width = image_tensor.shape
        aug_image = image_tensor.clone()
        patch_area = random.uniform(*area_ratio) * height * width
        patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
        patch_w  = int(np.sqrt(patch_area*patch_aspect))
        patch_h = int(np.sqrt(patch_area/patch_aspect))
        patch_left, patch_top = random.randint(0, height - patch_w), random.randint(0, width - patch_h)
        patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
    
        region = texture_tensor[:, :, patch_top:patch_bottom, patch_left:patch_right]
        
        aug_image[:, :, patch_top:patch_bottom, patch_left:patch_right] = region
        
        return aug_image
    
    def reconstruction(self, lightings):
        '''
        The reconstruction process
        :param y: the target image
        :param x: the input image
        :param seq: the sequence of denoising steps
        :param model: the UNet model
        :param x0_t: the prediction of x0 at time step t
        '''
        with torch.no_grad():
            # Convert images to latent space
            latents = self.image2latents(lightings)
            bsz = latents.shape[0]

            # Add noise to the latents according to the noise magnitude at each timestep
            _, timesteps, noisy_latents = self.forward_process_with_T(latents, self.timesteps_list[0].item()) 
            
            # Denoising Loop
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
            
            for t in self.timesteps_list:
                
                timesteps = torch.tensor([t.item()], device=self.device).repeat(bsz)
                timesteps = timesteps.long()

                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=self.encoder_hidden_states,
                ).sample
            
                sampled_latents = self.noise_scheduler.step(noise_pred.to(self.device), t.item(), noisy_latents.to(self.device)).prev_sample
            
            rec_images = self.latents2image(sampled_latents)
        return rec_images
    
    def create(self):
        
        for i, (lightings, _) in enumerate(tqdm(self.train_dataloader)):
            lightings = lightings.view(-1, 3, self.image_size, self.image_size).to(self.device)

            rec_image = self.reconstruction(lightings)
            augment_image = self.random_crop_and_paste(lightings, rec_image)
            display_3type_image(lightings, rec_image, augment_image, self.save_dir, i)
            

if __name__ == "__main__":

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device = {device}")
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, time)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    runner = CreateDefects(args=args, device=device)
    runner.create()