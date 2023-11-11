import torch
import numpy as np
import torch.nn.functional as F

from core.reconstruct_method import Reconstruct_Method

from utils.utils import t2np
from patchify import patchify
from utils.visualize_util import display_one_img, display_image
from diffusers import DDPMScheduler, UNet2DConditionModel, UniPCMultistepScheduler
from transformers import CLIPTextModel, AutoTokenizer

from core.models.network_util import Decom_Block
from core.models.controllora import  ControlLoRAModel

class Diffusion_Method(Reconstruct_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.diffusion_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.diffusion_id, subfolder="text_encoder")
        self.noise_scheduler = UniPCMultistepScheduler.from_pretrained(args.diffusion_id, subfolder="scheduler")
        self.num_inference_timesteps = int(len(self.noise_scheduler.timesteps) / args.step_size)
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)
        
        self.controllora: ControlLoRAModel
        self.controllora = ControlLoRAModel.from_unet(self.unet, lora_linear_rank=args.controllora_linear_rank, lora_conv2d_rank=args.controllora_conv2d_rank)
        self.controllora.load_state_dict(torch.load(args.controlnet_ckpt, map_location=self.device))
        self.controllora.tie_weights(self.unet)
        
        self.unet.to(self.device)
        self.text_encoder.to(self.device)
        self.controllora.to(self.device)
        
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.controllora.requires_grad_(False)
        self.controllora.eval()

        timesteps_list = self.noise_scheduler.timesteps[self.noise_scheduler.timesteps <= args.noise_intensity]
        print("Noise Intensity = ", timesteps_list[0].item())

        # Prepare text embedding
        self.encoder_hidden_states = self.get_text_embedding("", 6) # [6, 77, 768]
        
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

class ControlNet_Rec(Diffusion_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
    
    def predict(self, item, lightings, gt, label):

        lightings = lightings.to(self.device).view(-1, 3, self.image_size, self.image_size) # [6, 3, 256, 256]
        #fc_lightings = self.get_FC_lightings(lightings)
        nmaps = nmaps.to(self.device).repeat_interleave(6, dim=0) # [6, 3, 256, 256]
        
        # Convert images to latent space
        #latents = self.get_vae_latents(lightings)
        latents = self.autoencoder.image2latent(lightings)
        fc_latents = self.autoencoder.get_fc(latents)
        fu_latents = self.autoencoder.get_fu(latents)
        bsz = fc_latents.shape[0]

        # Add noise to the latents according to the noise magnitude at each timestep
        _, timesteps, noisy_latents = self.forward_process_with_T(fc_latents, timesteps_list[0].item()) 
        
        # Get CLIP embeddings
        encoder_hidden_states = self.get_text_embedding("", 6) # [6, 77, 768]
        #encoder_hidden_states = lighting_text_embedding # [6, 77, 768]

        condition_image = nmaps

        # Denoising Loop
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        for t in timesteps_list:

            timesteps = torch.tensor([t.item()], device=self.device).repeat(bsz)
            timesteps = timesteps.long()

            down_block_res_samples, mid_block_res_sample = self.controllora(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=condition_image,
                guess_mode=False,
                return_dict=False,
            )

            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[sample for sample in down_block_res_samples],
                mid_block_additional_residual=mid_block_res_sample
            ).sample
        return super().predict(item, lightings, gt, label)