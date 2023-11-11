import numpy as np
import torch
import torch.nn as nn
from core.models.network_util import Decom_Block
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDIMScheduler

class Autoencoder(nn.Module):
    def __init__(self, device, model_id="stabilityai/sd-vae-ft-ema"):
        super(Autoencoder, self).__init__()
        
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(model_id).to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        for name, param in self.vae.named_parameters():
            if 'decoder' in name:
                param.requires_grad = True
                
        self.decomp_block = Decom_Block(4).to(device)
    
    def randrec_forward(self, image):
        latents = self.image2latent(image)

        # random change fc fu reconstruction for bottleneck
        latents = self.decomp_block.prob_rand_forward(latents)
        rand_out = self.latent2image(latents)
        return rand_out
    
    def forward(self, image):
        x = self.image2latent(image)
        feat = self.decomp_block(x)
        out = self.latent2image(feat)
        return out
    
    def get_fc(self, latent):
        fc = self.decomp_block.get_fc(latent)
        return fc
    
    def get_fu(self, latent):
        fu = self.decomp_block.get_fu(latent)
        return fu
    
    def fuse_both(self, fc, fu):
        return self.decomp_block.fuse_both(fc, fu)
            
    def image2latent(self, image):
        with torch.no_grad():
            image = image.float() * 2.0 - 1.0
            # print(image.max())
            # print(image.min())
        latents = self.vae.encode(image).latent_dist.sample()
        latents = latents * 0.18215
        return latents

    def latent2image(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        return image.clamp(-1, 1)
    
    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False
    