import torch
import argparse
import numpy as np
import os
from torchvision import transforms
from tqdm import tqdm
from utils.au_pro_util import calculate_au_pro
from utils.visualize_util import *
from utils.utils import KNNGaussianBlur
from sklearn.metrics import roc_auc_score

# Diffusion model
from diffusers import DDIMScheduler
from core.models.unet_model import build_unet
from transformers import CLIPTextModel, AutoTokenizer
from diffusers import AutoencoderKL
from utils.ptp_utils import *

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LOCO_AD = {
    "breakfast_box" : { "good": 102, "SA":90, "LA":83},
    "juice_bottle" : { "good": 94, "SA":94, "LA":142},
    "pushpins" : { "good": 138, "SA":81, "LA":91},
    "screw_bag" : { "good": 122, "SA":82, "LA":137},
    "splicing_connectors" : { "good": 119, "SA":85, "LA":108},
}

class Base_Method():
    def __init__(self, args, cls_path):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_score()
        self.data_type = args.dataset_type
        self.blur = KNNGaussianBlur(4)
        self.criteria = torch.nn.MSELoss()    
        
        self.patch_lib = []
        self.nmap_patch_lib = []
        self.image_size = args.image_size
        
        self.cls_path = cls_path
        self.cls_rec_loss = 0.0
        self.reconstruct_path = os.path.join(cls_path, "Reconstruction")
        self.score_type = args.score_type
        self.viz = args.viz
        self.pdist = torch.nn.PairwiseDistance(p=2, eps= 1e-12)
        self.cos = torch.nn.CosineSimilarity()
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.resize = torch.nn.AdaptiveAvgPool2d((32, 32))
        self.image_transform = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        if not os.path.exists(self.reconstruct_path):
            os.makedirs(self.reconstruct_path)
            
    def initialize_score(self):
        self.image_list = list()
        
        self.image_labels = list()
        self.image_preds = list()
        self.pixel_labels = list()
        self.pixel_preds = list()
        self.predictions = []
        self.gts = []
        self.nmap_image_preds = []
        self.rgb_image_preds = []
        self.nmap_pixel_preds = []
        self.rgb_pixel_preds = []

    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):
        pass
    
    def alginment(self, lightings, nmap, text_prompt):
        pass
    
    def predict(self, item, lightings, gt, label):
        pass
    
    def cluster_training_data(self):
        pass

    def calculate_metrics(self, modality_name, cls_name=None):

        image_labels = np.stack(self.image_labels)
        pixel_labels = np.stack(self.pixel_labels)
         
        if modality_name == 'RGB' and self.rgb_image_preds:
            image_preds = np.stack(self.rgb_image_preds)
            pixel_preds = np.stack(self.rgb_pixel_preds)
        elif modality_name == 'Nmap' and self.nmap_image_preds:
            image_preds = np.stack(self.nmap_image_preds)
            pixel_preds = np.stack(self.nmap_pixel_preds)
        else:
            image_preds = np.stack(self.image_preds)
            pixel_preds = np.stack(self.pixel_preds)

        if modality_name == "SA":
            image_labels = np.concatenate((image_labels[:LOCO_AD[cls_name]["good"]], image_labels[-LOCO_AD[cls_name]["SA"]:]))
            image_preds = np.concatenate((image_preds[:LOCO_AD[cls_name]["good"]], image_preds[-LOCO_AD[cls_name]["SA"]:]))
            pixel_preds = np.concatenate((pixel_preds[:LOCO_AD[cls_name]["good"]], pixel_preds[-LOCO_AD[cls_name]["SA"]:]))
            pixel_labels = np.concatenate((pixel_labels[:LOCO_AD[cls_name]["good"]], pixel_labels[-LOCO_AD[cls_name]["SA"]:])).flatten()
        elif modality_name == "LA":
            image_labels = image_labels[:LOCO_AD[cls_name]["good"] + LOCO_AD[cls_name]["LA"]]
            image_preds = image_preds[:LOCO_AD[cls_name]["good"] + LOCO_AD[cls_name]["LA"]]
            pixel_preds = pixel_preds[:LOCO_AD[cls_name]["good"] + LOCO_AD[cls_name]["LA"]]
            pixel_labels = pixel_labels[:LOCO_AD[cls_name]["good"] + LOCO_AD[cls_name]["LA"]].flatten()
            
        flatten_pixel_labels = pixel_labels.flatten()
        flatten_pixel_preds = pixel_preds.flatten()
        
        gts = np.array(pixel_labels).reshape(-1, self.image_size, self.image_size)
        predictions =  pixel_preds.reshape(-1, self.image_size, self.image_size)

        self.image_rocauc = roc_auc_score(image_labels, image_preds)
        self.pixel_rocauc = roc_auc_score(flatten_pixel_labels, flatten_pixel_preds)
        self.au_pro, _ = calculate_au_pro(gts, predictions)
        return self.image_rocauc, self.pixel_rocauc, self.au_pro
    
    def get_rec_loss(self):
        return self.cls_rec_loss
    
    def visualizae_heatmap(self, modality_name, cls_name=None):
        self.pixel_labels = np.stack(self.pixel_labels)
        cls_path = os.path.join(self.cls_path, modality_name)
        if modality_name == 'RGB' and self.rgb_pixel_preds:
            pixel_preds = np.stack(self.rgb_pixel_preds)
        elif modality_name == 'Nmap' and self.nmap_pixel_preds:
            pixel_preds = np.stack(self.nmap_pixel_preds)
        elif (modality_name == 'RGB+Nmap' or modality_name == 'ALL') and self.pixel_preds:
            pixel_preds = np.stack(self.pixel_preds)
        else:
            pixel_preds = np.stack(self.pixel_preds)
            if modality_name == 'SA':
                pixel_preds = np.concatenate((pixel_preds[:LOCO_AD[cls_name]["good"]], pixel_preds[-LOCO_AD[cls_name]["SA"]:]))
                score_map = pixel_preds.reshape(-1, self.image_size, self.image_size)
                gt_mask = np.squeeze(np.array(np.concatenate((self.pixel_labels[:LOCO_AD[cls_name]["good"]], self.pixel_labels[-LOCO_AD[cls_name]["SA"]:])), dtype=np.bool_))
                image_list = np.concatenate((self.image_list[:LOCO_AD[cls_name]["good"]], self.image_list[-LOCO_AD[cls_name]["SA"]:]))
                image_labels = np.concatenate((self.image_labels[:LOCO_AD[cls_name]["good"]], self.image_labels[-LOCO_AD[cls_name]["SA"]:]))
                image_preds = np.concatenate((self.image_preds[:LOCO_AD[cls_name]["good"]], self.image_preds[-LOCO_AD[cls_name]["SA"]:]))
                visualization(image_list, image_labels, image_preds, gt_mask, score_map, cls_path)
            elif modality_name == 'LA':
                pixel_preds = pixel_preds[:LOCO_AD[cls_name]["good"] + LOCO_AD[cls_name]["LA"]]
                score_map = pixel_preds.reshape(-1, self.image_size, self.image_size)
                gt_mask = np.squeeze(np.array(self.pixel_labels[:LOCO_AD[cls_name]["good"] + LOCO_AD[cls_name]["LA"]], dtype=np.bool_))
                image_list = self.image_list[:LOCO_AD[cls_name]["good"] + LOCO_AD[cls_name]["LA"]]
                image_labels =self.image_labels[:LOCO_AD[cls_name]["good"] + LOCO_AD[cls_name]["LA"]]
                image_preds = self.image_preds[:LOCO_AD[cls_name]["good"] + LOCO_AD[cls_name]["LA"]]
                visualization(image_list, image_labels, image_preds, gt_mask, score_map, cls_path)
            

        score_map = pixel_preds.reshape(-1, self.image_size, self.image_size)
        gt_mask = np.squeeze(np.array(self.pixel_labels, dtype=np.bool_))
        # visualization(self.image_list, self.image_labels, self.image_preds, gt_mask, score_map, cls_path)
        
        # score_map_z = z_score_normalization(score_map.flatten())
        # score_map_mn = min_max_normalization(score_map.flatten())
        # visualize_perpixel_distribute(score_map.flatten(), gt_mask, cls_path, "distribution_wonormalize")
        # visualize_perpixel_distribute(score_map_z, gt_mask, cls_path, "distribution_of_zscore")
        # visualize_perpixel_distribute(score_map_mn, gt_mask, cls_path, "distribution_of_minmax")
        rgb_s = np.array(self.image_preds)
        label = np.array(self.image_labels)
        
        visualize_image_s_distribute(rgb_s, label, cls_path)
        
        
    def cal_alignment(self):
        # nmap distribution
        nmap = np.array(self.nmap_pixel_preds)
        non_zero_indice = np.nonzero(nmap)
        non_zero_nmap = nmap[non_zero_indice]
        nmap_mean = np.mean(non_zero_nmap)
        nmap_std = np.std(non_zero_nmap)
        nmap_lower = nmap_mean - 3 * nmap_std
        nmap_upper = nmap_mean + 3 * nmap_std
        # RGB distribution
        rgb_map = np.array(self.rgb_pixel_preds)
        non_zero_indice = np.nonzero(rgb_map)
        non_zero_rgb_map = rgb_map[non_zero_indice]
        rgb_mean = np.mean(non_zero_rgb_map)
        rgb_std = np.std(non_zero_rgb_map)
        rgb_lower = rgb_mean - 3 * rgb_std
        rgb_upper = rgb_mean + 3 * rgb_std
        
        self.weight = (nmap_upper - nmap_lower) / (rgb_upper - rgb_lower)
        self.bias = nmap_lower - self.weight * rgb_lower
        print("weight:", self.weight)
        print("bias:", self.bias)
        self.nmap_image_preds = []
        self.rgb_image_preds = []
        self.nmap_pixel_preds = []
        self.rgb_pixel_preds = []
    
class DDIM_Method(Base_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.diffusion_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.diffusion_id, subfolder="text_encoder")
        self.noise_scheduler = DDIMScheduler.from_pretrained(args.diffusion_id, subfolder="scheduler")
        self.num_inference_timesteps = int(len(self.noise_scheduler.timesteps) / args.step_size)
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)
        self.timesteps_list = self.noise_scheduler.timesteps[self.noise_scheduler.timesteps <= max(args.noise_intensity)]

        
        self.text_encoder.to(self.device)
        self.step_size = args.step_size
        print("num_inference_timesteps")
        print("ddim loop steps:", len(self.timesteps_list))
        print("Noise Intensity = ", self.timesteps_list)

        self.unet = build_unet(args)
        self.unet.to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            args.diffusion_id,
            subfolder="vae",
            # revision=args.revision,
            torch_dtype=torch.float32
        ).to(self.device)
        
        if os.path.isfile(args.load_unet_ckpt):
            self.unet.load_state_dict(torch.load(args.load_unet_ckpt, map_location=self.device))
            print("Load Diffusion Model Checkpoint!!")
        
        if os.path.isfile(args.load_vae_ckpt):
            checkpoint_dict = torch.load(args.load_vae_ckpt, map_location=self.device)
            ## Load VAE checkpoint  
            if checkpoint_dict['vae'] is not None:
                print("Load vae checkpoints!!")
                self.vae.load_state_dict(checkpoint_dict['vae'])
        
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.unet.eval()
        self.text_encoder.eval()
          
        # Prepare text embedding
        self.uncond_embeddings = self.get_text_embedding("", 6) # [6, 77, 768]

        self.mul_timesteps = args.noise_intensity
        self.reweight = args.reweight
        self.feature_layers = args.feature_layers
        self.topk = args.topk
    
    @torch.no_grad()
    def image2latents(self, x):
        x = x * 2.0 - 1.0
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * 0.18215
        return latents
    
    @torch.no_grad()
    def latents2image(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image   
    
    @torch.no_grad()     
    def forward_process_with_T(self, latents, T):
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        timesteps = torch.tensor([T], device=self.device).repeat(bsz)
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        return noise, timesteps, noisy_latents

    @torch.no_grad()
    def get_text_embedding(self, text_prompt, bsz):
        tok = self.tokenizer(text_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embedding = self.text_encoder(tok.input_ids.to(self.device))[0].repeat((bsz, 1, 1))
        return text_embedding
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.noise_scheduler.config.num_train_timesteps // self.noise_scheduler.num_inference_steps
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.noise_scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.noise_scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.noise_scheduler.config.num_train_timesteps // self.noise_scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.noise_scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.noise_scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.unet(latents, t, encoder_hidden_states=context)['sample']
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.guidance_scale
        noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents