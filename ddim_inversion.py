from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import argparse
import abc
# import ptp_utils
# import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
from core.data import test_lightings_loader
from utils.ptp_utils import *
from utils.visualize_util import display_one_img, display_image, display_mean_fusion

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--data_path', default="/mnt/home_6T/public/jayliu0313/datasets/Eyecandies/", type=str)
parser.add_argument("--load_unet_ckpt", default="/mnt/home_6T/public/jayliu0313/check_point/Diffusion_ckpt/unet_InputImage_TextNull_woTrainVae/best_unet.pth")
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument("--workers", default=8)

args = parser.parse_args()
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ''
LOW_RESOURCE = False
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)

ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN, scheduler=scheduler).to(device)
if args.load_unet_ckpt is not None:
    print("Load Unet Checkpoint from", args.load_unet_ckpt)
    ldm_stable.unet.load_state_dict(torch.load(args.load_unet_ckpt, map_location=device))
# try:
#     ldm_stable.disable_xformers_memory_efficient_attention()
# except AttributeError:
#     print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer

# def load_512(image_path, left=0, right=0, top=0, bottom=0):
#     if type(image_path) is str:
#         image = np.array(Image.open(image_path))[:, :, :3]
#     else:
#         image = image_path
#     h, w, c = image.shape
#     left = min(left, w-1)
#     right = min(right, w - left - 1)
#     top = min(top, h - left - 1)
#     bottom = min(bottom, h - top - 1)
#     image = image[top:h-bottom, left:w-right]
#     h, w, c = image.shape
#     if h < w:
#         offset = (w - h) // 2
#         image = image[:, offset:offset + h]
#     elif w < h:
#         offset = (h - w) // 2
#         image = image[offset:offset + w]
#     image = np.array(Image.fromarray(image).resize((512, 512)))
#     return image



class NullInversion:

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            image = image.float() * 2.0 - 1
            # image = image.permute(2, 0, 1).unsqueeze(0).to(device)
            latents = self.model.vae.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
        return latents
    
    @torch.no_grad()
    def get_text_embedding(self, text_prompt, bsz):
        tok = self.model.tokenizer(text_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embedding = self.model.text_encoder(tok.input_ids.to(device).repeat(bsz,1))[0]
        self.context = text_embedding
    
    @torch.no_grad()
    def ddim_loop(self, latent):
        cond_embeddings = self.context
        all_latent = [latent]
        latent = latent.clone().detach()
        print(self.model.scheduler.timesteps)
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents


    def invert(self, image, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.get_text_embedding(prompt, args.batch_size * 6)
        # ptp_utils.register_attention_control(self.model, None)
        # image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image)
        # if verbose:
        #     print("Null-text optimization...")
        # uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return image, image_rec, ddim_latents#, uncond_embeddings


    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None

    @torch.no_grad()
    def text2image_ldm_stable(
        self,
        noisy_latents,
        num_inference_steps: int = 50,
        start_time=50,
    ):
        bsz = noisy_latents.shape[0]
        # latent, latents = init_latent(latent, model, height, width, generator, batch_size)
        self.model.scheduler.set_timesteps(num_inference_steps)
        for i, t in enumerate(tqdm(self.model.scheduler.timesteps[-start_time:])):
            timesteps = torch.tensor([t.item()], device=device).repeat(bsz)
            timesteps = timesteps.long()
            noise_pred = self.model.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=self.context,
            ).sample
            noisy_latents = self.model.scheduler.step(noise_pred, t.item(), noisy_latents)["prev_sample"]
        image = self.latent2image(noisy_latents)
        return image

null_inversion = NullInversion(ldm_stable)

classes = [
        'CandyCane',
        'ChocolateCookie',
        # 'ChocolatePraline',
        # 'Confetto',
        'GummyBear',
        # 'HazelnutTruffle',
        # 'LicoriceSandwich',
        # 'Lollipop',
        # 'Marshmallow',
        # 'PeppermintCandy'
        ]
prompt = ""
for cls in classes:
    test_loader = test_lightings_loader(args, cls, "test")
    
    for i, ((images, nmap), gt, label) in enumerate(tqdm(test_loader)):
        images = images.reshape(-1, 3, args.image_size, args.image_size).to(device)

        gt_image, image_rec, ddim_latents = null_inversion.invert(images, prompt, offsets=(0,0,200,0), verbose=True)
        sampled_images = null_inversion.text2image_ldm_stable(ddim_latents[-1], NUM_DDIM_STEPS)
        # display_image(gt_image, sampled_images, "./visualize_sampledImage/", i)
