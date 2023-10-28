import numpy as np
import torch
import torch.nn as nn
from core.models.network_util import Decom_Block
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDIMScheduler
from transformers import CLIPFeatureExtractor, CLIPTokenizer, CLIPProcessor, CLIPVisionModel
from PIL import Image

class Autoencoder(nn.Module):
    def __init__(self, device, model_id="stabilityai/sd-vae-ft-ema"):
        super(Autoencoder, self).__init__()
        
        # self.vae = AutoencoderKL.from_pretrained(model_id).to(device)
        # self.vae.requires_grad_(False)
        # self.vae.eval()
        # scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        # MY_TOKEN = ''
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(model_id).to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        for name, param in self.vae.named_parameters():
            if 'decoder' in name:
                param.requires_grad = True
                
        self.decomp_block = Decom_Block(4).to(device)

        # self.vae.eval()
        # self.vae.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        # self.criteria = torch.nn.MSELoss()
    
    def rec_randrec_forward(self, image):
        x = self.image2latent(image)
        
        # normal reconstruction
        feat = self.decomp_block(x)
        out = self.latent2image(feat)
        # random change fc fu reconstruction for bottleneck
        rand_feat = self.decomp_block.rand_forward(x)
        rand_out = self.latent2image(rand_feat)
        return out, rand_out
    
    def randrec_forward(self, image):
        latents = self.image2latent(image)
        
        # normal reconstruction
        # feat = self.decomp_block(x)
        # out = self.latent2image(feat)
        # random change fc fu reconstruction for bottleneck
        latents = self.decomp_block.prob_rand_forward(latents)
        rand_out = self.latent2image(latents)
        return rand_out
    
    def forward(self, image):
        x = self.image2latent(image)
        feat = self.decomp_block(x)
        out = self.latent2image(feat)
        return out
        
    def image2latent(self, image):
        # with torch.no_grad():
            # image = torch.from_numpy(image).float() / 127.5 - 1
            # image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
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
    
# class Autoencoder(nn.Module):
#     def __init__(self, device, latent_dim=256):
#         super(Autoencoder, self).__init__()
#         # self.feature_extractor = RGB_Extractor(device)
#         # self.feature_extractor.freeze_parameters(layers=[], freeze_bn=True)
#         # feature_dims = [512, 1024, 2048]
#         # in_dim = sum(feature_dims)
#         # self.scale_factors = [
#         #     in_stride / 128 for in_stride in feature_dims
#         # ]  # for resize
#         # self.upsample_list = [
#         #     nn.UpsamplingBilinear2d(scale_factor=scale_factor)
#         #     for scale_factor in self.scale_factors
#         # ]
#         # self.feature_mapping = nn.Conv2d(in_dim, latent_dim, 3, 1, 1)
#         # self.decomp = Decom_Block(latent_dim)
#         # self.feature_mappings = [
#         #     nn.Conv2d()
#         # ]
#         # self.decoder = ResNet_Decoder(256, 16)
#         vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch.float16).to("cuda")
#         latent_map = vae.encode(img_tensor).latent_dist.sample() * 0.18215
        
#     def forward(self, x):
#         rgb_feature_maps = self.feature_extractor(x)
        
#         for i in rgb_feature_maps:
#             print(i.shape)
#         feature_list = []
        
#         # resize & concatenate
#         # for i in range(len(rgb_feature_maps)):
#         #     upsample = self.upsample_list[i]
#         #     feature_resize = upsample(rgb_feature_maps[i])
#         #     print(feature_resize.shape)
#         #     feature_list.append(feature_resize)
            
#         # cat_feature = torch.cat(feature_list, dim=1)
#         # align_feature = self.feature_mapping(cat_feature)
#         # fuse_feature = self.decomp(align_feature)
#         # out = self.decoder(fuse_feature)
#         return out