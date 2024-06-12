import os
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


from core.data import *
from transformers import CLIPTextModel, AutoTokenizer
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from core.models.unet_model import build_unet
from core.models.controllora import ControlLoRAModel

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--data_path', default="/mnt/home_6T/public/samchu0218/Datasets/mvtec3d_preprocessing/", type=str)
# "/mnt/home_6T/public/jayliu0313/datasets/Eyecandies/"
# "/mnt/home_6T/public/samchu0218/Datasets/mvtec3d_preprocessing/"
#"/mnt/home_6T/public/samchu0218/Raw_Datasets/MVTec_AD/MVTec_Loco/"
parser.add_argument('--ckpt_path', default="./checkpoints/controlnet_model/mvtec3d_InfoNCE/") # 
parser.add_argument('--load_vae_ckpt', default=None)
parser.add_argument('--load_unet_ckpt', default="/home/samchu0218/Multi_Lightings/checkpoints/unet_model/MVTec3D/epoch10_unet.pth")
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--dataset_type', default="eyecandies", help="eyecandies, mvtec3d")
parser.add_argument('--use_floss', default=True, type=bool)
# parser.add_argument('--')

# Model Setup
parser.add_argument("--diffusion_id", type=str, default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--revision", type=str, default="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c")
parser.add_argument("--controllora_linear_rank",type=int, default=4)
parser.add_argument("--controllora_conv2d_rank",type=int,default=0)

# Training Setup
parser.add_argument("--learning_rate", default=5e-6)
parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
parser.add_argument("--workers", default=4)
parser.add_argument('--CUDA', type=int, default=0, help="choose the device of CUDA")
parser.add_argument("--lr_scheduler", type=str, default="constant", help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'' "constant", "constant_with_warmup"]'),)
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument('--epoch', default=0, type=int, help="Which epoch to start training at")
parser.add_argument("--num_train_epochs", type=int, default=100)
parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
parser.add_argument("--save_epoch", type=int, default=3)


def export_loss(save_path, loss_list):
    epoch_list = range(len(loss_list))
    plt.rcParams.update({'font.size': 30})
    plt.title('Training Loss Curve') # set the title of graph
    plt.figure(figsize=(20, 15))
    plt.plot(epoch_list, loss_list, color='b')
    plt.xticks(np.arange(0, len(epoch_list)+1, 50))
    plt.xlabel('Epoch') # set the title of x axis
    plt.ylabel('Loss')
    plt.savefig(save_path)
    plt.clf()
    plt.cla()
    plt.close("all")

def compute_mean_feature_loss(model_output):
    # Extract Unet Feature Map
    unet_f_layer0 = model_output['up_ft'][0]
    B, C, H, W = unet_f_layer0.shape
    RGB_unet_f_layer0  = unet_f_layer0[:args.batch_size * 6]
    Nmap_unet_f_layer0 = unet_f_layer0[args.batch_size * 6:]

    mean_RGBunet_f_layer0 = torch.mean(RGB_unet_f_layer0.view(-1, 6, C, H, W), dim=1)
    mean_RGBunet_f_layer0 = mean_RGBunet_f_layer0.repeat_interleave(6, dim=0)
    mean_Nmapunet_f_layer0 = torch.mean(Nmap_unet_f_layer0.view(-1, 6, C, H, W), dim=1)
    mean_Nmapunet_f_layer0 = mean_Nmapunet_f_layer0.repeat_interleave(6, dim=0)

    unet_f_layer1 = model_output['up_ft'][1]
    B, C, H, W = unet_f_layer1.shape
    RGB_unet_f_layer1  = unet_f_layer1[:args.batch_size * 6]
    Nmap_unet_f_layer1 = unet_f_layer1[args.batch_size * 6:]

    mean_RGBunet_f_layer1 = torch.mean(RGB_unet_f_layer1.view(-1, 6, C, H, W), dim=1)
    mean_RGBunet_f_layer1 = mean_RGBunet_f_layer1.repeat_interleave(6, dim=0)
    mean_Nmapunet_f_layer1 = torch.mean(Nmap_unet_f_layer1.view(-1, 6, C, H, W), dim=1)
    mean_Nmapunet_f_layer1 = mean_Nmapunet_f_layer1.repeat_interleave(6, dim=0)
    
    unet_f_layer2 = model_output['up_ft'][2]
    B, C, H, W = unet_f_layer2.shape
    RGB_unet_f_layer2  = unet_f_layer2[:args.batch_size * 6]
    Nmap_unet_f_layer2 = unet_f_layer2[args.batch_size * 6:]

    mean_RGBunet_f_layer2 = torch.mean(RGB_unet_f_layer2.view(-1, 6, C, H, W), dim=1)
    mean_RGBunet_f_layer2 = mean_RGBunet_f_layer2.repeat_interleave(6, dim=0)
    mean_Nmapunet_f_layer2 = torch.mean(Nmap_unet_f_layer2.view(-1, 6, C, H, W), dim=1)
    mean_Nmapunet_f_layer2 = mean_Nmapunet_f_layer2.repeat_interleave(6, dim=0)

    unet_f_layer3 = model_output['up_ft'][3]
    B, C, H, W = unet_f_layer3.shape
    RGB_unet_f_layer3  = unet_f_layer3[:args.batch_size * 6]
    Nmap_unet_f_layer3 = unet_f_layer3[args.batch_size * 6:]
    mean_RGBunet_f_layer3 = torch.mean(RGB_unet_f_layer3.view(-1, 6, C, H, W), dim=1)
    mean_RGBunet_f_layer3 = mean_RGBunet_f_layer3.repeat_interleave(6, dim=0)
    mean_Nmapunet_f_layer3 = torch.mean(Nmap_unet_f_layer3.view(-1, 6, C, H, W), dim=1)
    mean_Nmapunet_f_layer3 = mean_Nmapunet_f_layer3.repeat_interleave(6, dim=0)

    # Compute loss and optimize model parameter
    RGBfeature_loss = F.l1_loss(mean_RGBunet_f_layer0, RGB_unet_f_layer0, reduction="mean")
    RGBfeature_loss += F.l1_loss(mean_RGBunet_f_layer1, RGB_unet_f_layer1, reduction="mean")
    RGBfeature_loss += F.l1_loss(mean_RGBunet_f_layer2, RGB_unet_f_layer2, reduction="mean")
    RGBfeature_loss += F.l1_loss(mean_RGBunet_f_layer3, RGB_unet_f_layer3, reduction="mean")

    Nmapfeature_loss = F.l1_loss(mean_Nmapunet_f_layer0, Nmap_unet_f_layer0, reduction="mean")
    Nmapfeature_loss += F.l1_loss(mean_Nmapunet_f_layer1, Nmap_unet_f_layer1, reduction="mean")
    Nmapfeature_loss += F.l1_loss(mean_Nmapunet_f_layer2, Nmap_unet_f_layer2, reduction="mean")
    Nmapfeature_loss += F.l1_loss(mean_Nmapunet_f_layer3, Nmap_unet_f_layer3, reduction="mean")
    feature_loss = RGBfeature_loss + Nmapfeature_loss
    
    return feature_loss

def compute_diff_modality_loss(model_output):
    unet_f_layer0 = model_output['up_ft'][0]
    RGB_unet_f_layer0  = unet_f_layer0[:args.batch_size]
    Nmap_unet_f_layer0 = unet_f_layer0[args.batch_size:]


    unet_f_layer1 = model_output['up_ft'][1]
    RGB_unet_f_layer1  = unet_f_layer1[:args.batch_size]
    Nmap_unet_f_layer1 = unet_f_layer1[args.batch_size:]

    
    unet_f_layer2 = model_output['up_ft'][2]
    RGB_unet_f_layer2  = unet_f_layer2[:args.batch_size]
    Nmap_unet_f_layer2 = unet_f_layer2[args.batch_size:]



    unet_f_layer3 = model_output['up_ft'][3]
    RGB_unet_f_layer3  = unet_f_layer3[:args.batch_size]
    Nmap_unet_f_layer3 = unet_f_layer3[args.batch_size:]


    # Compute loss and optimize model parameter
    feature_loss = F.l1_loss(RGB_unet_f_layer0, Nmap_unet_f_layer0, reduction="mean")
    feature_loss += F.l1_loss(RGB_unet_f_layer1, Nmap_unet_f_layer1, reduction="mean")
    feature_loss += F.l1_loss(RGB_unet_f_layer2, Nmap_unet_f_layer2, reduction="mean")
    feature_loss += F.l1_loss(RGB_unet_f_layer3, Nmap_unet_f_layer3, reduction="mean")
    return feature_loss    

def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.T = 0.5

    def info_nce(self, query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        B, C, H, W = query.shape
        query = query.permute(0, 2, 3, 1).reshape(B*H*W, C)
        positive_key = positive_key.permute(0, 2, 3, 1).reshape(B*H*W, C)
        # Check input dimensionality.
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        if positive_key.dim() != 2:
            raise ValueError('<positive_key> must have 2 dimensions.')
        if negative_keys is not None:
            if negative_mode == 'unpaired' and negative_keys.dim() != 2:
                raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
            if negative_mode == 'paired' and negative_keys.dim() != 3:
                raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

        # Check matching number of samples.
        if len(query) != len(positive_key):
            raise ValueError('<query> and <positive_key> must must have the same number of samples.')
        if negative_keys is not None:
            if negative_mode == 'paired' and len(query) != len(negative_keys):
                raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

        # Embedding vectors should have same number of components.
        if query.shape[-1] != positive_key.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
        if negative_keys is not None:
            if query.shape[-1] != negative_keys.shape[-1]:
                raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

        # Normalize to unit vectors
        query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
        if negative_keys is not None:
            # Explicit negative keys

            # Cosine between positive pairs
            positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

            if negative_mode == 'unpaired':
                # Cosine between all query-negative combinations
                negative_logits = query @ transpose(negative_keys)

            elif negative_mode == 'paired':
                query = query.unsqueeze(1)
                negative_logits = query @ transpose(negative_keys)
                negative_logits = negative_logits.squeeze(1)

            # First index in last dimension are the positive samples
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        else:
            # Negative keys are implicitly off-diagonal positive keys.

            # Cosine between all combinations
            logits = query @ transpose(positive_key)

            # Positive keys are the entries on the diagonal
            labels = torch.arange(len(query), device=query.device)

        return F.cross_entropy(logits / temperature, labels, reduction=reduction)


    def forward(self, model_output):
        unet_f_layer0 = model_output['up_ft'][0]
        RGB_unet_f_layer0  = unet_f_layer0[:args.batch_size]
        Nmap_unet_f_layer0 = unet_f_layer0[args.batch_size:]
        loss = self.info_nce(RGB_unet_f_layer0, Nmap_unet_f_layer0)
        
        unet_f_layer1 = model_output['up_ft'][1]
        RGB_unet_f_layer1  = unet_f_layer1[:args.batch_size]
        Nmap_unet_f_layer1 = unet_f_layer1[args.batch_size:]
        loss += self.info_nce(RGB_unet_f_layer1, Nmap_unet_f_layer1)
        
        unet_f_layer2 = model_output['up_ft'][2]
        RGB_unet_f_layer2  = unet_f_layer2[:args.batch_size]
        Nmap_unet_f_layer2 = unet_f_layer2[args.batch_size:]
        loss += self.info_nce(RGB_unet_f_layer2, Nmap_unet_f_layer2)

        unet_f_layer3 = model_output['up_ft'][3]
        RGB_unet_f_layer3  = unet_f_layer3[:args.batch_size]
        Nmap_unet_f_layer3 = unet_f_layer3[args.batch_size:]
        loss += self.info_nce(RGB_unet_f_layer3, Nmap_unet_f_layer3)
        return loss
    
    # def compute_loss(self, feature_map1, feature_map2):
    #     B, C, H, W = feature_map1.shape
    #     # Flatten the feature maps
    #     feature_map1_flat = feature_map1.permute(0, 2, 3, 1).reshape(B*H*W, C)
    #     feature_map2_flat = feature_map2.permute(0, 2, 3, 1).reshape(B*H*W, C)
    #     q = nn.functional.normalize(feature_map1_flat, dim=1)
    #     k = nn.functional.normalize(feature_map2_flat, dim=1)
    #     # Compute cosine similarity
    #     similarities = torch.einsum('nc,mc->nm', [q, k]) / self.T
    #     N = similarities.shape[0]
    #     labels = torch.arange(N, dtype=torch.long).cuda()
    #     loss = nn.CrossEntropyLoss()(similarities, labels) * (2 * self.T)
    #     return loss

class TrainUnet():
    def __init__(self, args, device):

        self.device = device
        self.bs = args.batch_size
        self.image_size = args.image_size
        self.num_train_epochs = args.num_train_epochs
        self.save_epoch = args.save_epoch
        self.train_log_file = open(osp.join(args.ckpt_path, "training_log.txt"), "a", 1)
        self.val_log_file = open(osp.join(args.ckpt_path, "val_log.txt"), "a", 1)
        self.use_floss = args.use_floss
        self.dataset_type = args.dataset_type
        # Load training and validation data
        if args.dataset_type == "eyecandies":
            self.train_dataloader = train_lightings_loader(args)
            self.val_dataloader = val_lightings_loader(args)
        elif args.dataset_type == "mvtec3d":
            self.train_dataloader = mvtec3D_train_loader(args)
            self.val_dataloader = mvtec3D_val_loader(args)
            self.contrastive = ContrastiveLoss()
        elif args.dataset_type == "mvtecloco":
            self.train_dataloader = mvtecLoco_train_loader(args)
            self.val_dataloader = mvtecLoco_val_loader(args)

        # Create Model
        self.tokenizer = AutoTokenizer.from_pretrained(args.diffusion_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.diffusion_id, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.diffusion_id, subfolder="scheduler")

        self.vae = AutoencoderKL.from_pretrained(
            args.diffusion_id,
            subfolder="vae",
            revision=args.revision,
        ).to(self.device)
        
        self.unet = build_unet(args)
        
        if os.path.isfile(args.load_unet_ckpt):
            self.unet.load_state_dict(torch.load(args.load_unet_ckpt, map_location=self.device))
            print("Load Diffusion Unet Checkpoint!")
        self.controllora = ControlLoRAModel.from_unet(self.unet, lora_linear_rank=args.controllora_linear_rank, lora_conv2d_rank=args.controllora_conv2d_rank)

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.controllora.train()

        self.vae.to(self.device)
        self.unet.to(self.device)
        self.text_encoder.to(self.device)
        self.controllora.to(self.device)

        # Optimizer creation 
        self.optimizer = torch.optim.AdamW(
            self.controllora.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_dataloader) * args.num_train_epochs,
            num_cycles=1,
            power=1.0,
        )

    def image2latents(self, x):
        x = x * 2.0 - 1.0
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * 0.18215
        return latents

    def latents2image(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        return image.clamp(-1, 1)

    def forward_process(self, x_0):
        noise = torch.randn_like(x_0) # Sample noise that we'll add to the latents
        bsz = x_0.shape[0]

        timestep = torch.randint(1, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device) # Sample a random timestep for each image
        timestep = timestep.long()
        x_t = self.noise_scheduler.add_noise(x_0, noise, timestep) # Corrupt image
        return noise, timestep, x_t

    def get_text_embedding(self, text_prompt):
        with torch.no_grad():
            tok = self.tokenizer(text_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embedding = self.text_encoder(tok.input_ids.to(self.device))[0]
        return text_embedding

    def log_validation(self):
        val_loss = 0.0
        val_noise_loss = 0.0
        val_feature_loss = 0.0
        
        for lightings, nmaps, text_prompt in tqdm(self.val_dataloader, desc="Validation"):

            with torch.no_grad():
                self.optimizer.zero_grad()
                lightings = lightings.to(self.device) # [bs, 6, 3, 256, 256]
                nmaps = nmaps.to(self.device)         # [bs, 6, 3, 256, 256]
                # Get text embedding from CLIP
                text_emb = self.get_text_embedding(text_prompt)  # [bs    , 7, 768]
                
                lightings = lightings.view(-1, 3, self.image_size, self.image_size) # [bs * 6, 3, 256, 256]

                # Convert images to latent space
                rgb_latents = self.image2latents(lightings)        # [bs * 6, 4, 32, 32]
                nmap_latents = self.image2latents(nmaps)    # [bs * 6, 4, 32, 32]
                
                if self.dataset_type == "eyecandies":
                    repeat_nmaps = nmaps.repeat_interleave(6, dim=0)                    # [bs * 6, 3, 256, 256]
                    text_embs = text_emb.repeat_interleave(6, dim=0) # [bs * 6, 7, 768]
                    nmap_latents = nmap_latents.repeat_interleave(6, dim=0) 
                else:
                    repeat_nmaps = nmaps
                    text_embs = text_emb
                    nmap_latents = nmap_latents
                    
                encoder_hidden_states = torch.cat((text_embs, text_embs), dim=0) # [bs * 12, 77, 768]
                
                input_latent = torch.cat((rgb_latents, nmap_latents), dim=0) # [bs * 12, 4, 32, 32]
                
                # Add noise to the latents according to the noise magnitude at each timestep
                noise, timesteps, noisy_latents = self.forward_process(input_latent)

                # Training ControlNet
                condition_image = torch.cat((repeat_nmaps, lightings), dim=0) # [bs * 7, 3, 256, 256]

                down_block_res_samples, mid_block_res_sample = self.controllora(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=condition_image,
                    return_dict=False,
                )

                # Predict the noise from Unet
                model_output = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[sample for sample in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample
                )
                pred_noise = model_output['sample']
                noise_loss = F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")
                if self.dataset_type == "eyecandies" and self.use_floss:
                    feature_loss = compute_mean_feature_loss(model_output)
                    loss = noise_loss + 0.01 * feature_loss
                    val_noise_loss += noise_loss.item()
                    val_feature_loss += feature_loss.item()
                if self.dataset_type == "mvtec3d" and self.use_floss:
                    feature_loss = self.contrastive(model_output)
                    loss = noise_loss + 0.001 * feature_loss
                    val_noise_loss += noise_loss.item()
                    val_feature_loss += feature_loss.item()
                else:
                    loss = noise_loss
                val_loss += loss.item()


        val_loss /= len(self.val_dataloader)
        val_noise_loss /= len(self.val_dataloader)
        val_feature_loss /= len(self.val_dataloader)
        print('Validation Loss: {:.6f}, Noise loss: {:.6f}, Feature loss:{:.6f}'.format(val_loss, val_noise_loss, val_feature_loss))
        self.val_log_file.write('Validation Loss: {:.6f}, Noise loss: {:.6f}, Feature loss:{:.6f}\n'.format(val_loss, val_noise_loss, val_feature_loss))
        return val_loss

    #####################################################################
    #                            Start Training                         #
    #####################################################################
    def train(self):
        
        loss_list = []
        noise_loss_list = []
        feature_loss_list = []
        val_loss_list = []
        val_best_loss = float('inf')

        for epoch in range(self.num_train_epochs):

            epoch_loss = 0.0
            epoch_noise_loss = 0.0
            epoch_feature_loss = 0.0

            for lightings, nmaps, text_prompt in tqdm(self.train_dataloader, desc="Training"):

                self.optimizer.zero_grad()
                lightings = lightings.to(self.device) # [bs, 6, 3, 256, 256]
                nmaps = nmaps.to(self.device)         # [bs, 6, 3, 256, 256]
                # Get text embedding from CLIP
                text_emb = self.get_text_embedding(text_prompt)  # [bs    , 7, 768]
                
                lightings = lightings.view(-1, 3, self.image_size, self.image_size) # [bs * 6, 3, 256, 256]

                # Convert images to latent space
                rgb_latents = self.image2latents(lightings)        # [bs * 6, 4, 32, 32]
                nmap_latents = self.image2latents(nmaps)    # [bs * 6, 4, 32, 32]
                
                if self.dataset_type == "eyecandies":
                    repeat_nmaps = nmaps.repeat_interleave(6, dim=0)                    # [bs * 6, 3, 256, 256]
                    text_embs = text_emb.repeat_interleave(6, dim=0) # [bs * 6, 7, 768]
                    nmap_latents = nmap_latents.repeat_interleave(6, dim=0) 
                else:
                    repeat_nmaps = nmaps
                    text_embs = text_emb
                    nmap_latents = nmap_latents
                    
                encoder_hidden_states = torch.cat((text_embs, text_embs), dim=0) # [bs * 12, 77, 768]
                
                input_latent = torch.cat((rgb_latents, nmap_latents), dim=0) # [bs * 12, 4, 32, 32]
                
                # Add noise to the latents according to the noise magnitude at each timestep
                noise, timesteps, noisy_latents = self.forward_process(input_latent)

                # Training ControlNet
                condition_image = torch.cat((repeat_nmaps, lightings), dim=0) # [bs * 7, 3, 256, 256]

                down_block_res_samples, mid_block_res_sample = self.controllora(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=condition_image,
                    return_dict=False,
                )

                # Predict the noise from Unet
                model_output = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[sample for sample in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample
                )
                
                pred_noise = model_output['sample']
                noise_loss = F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")
                
                if self.dataset_type == "eyecandies" and self.use_floss:
                    feature_loss = compute_mean_feature_loss(model_output)
                    loss = noise_loss + 0.01 * feature_loss
                    epoch_noise_loss += noise_loss.item()
                    epoch_feature_loss += feature_loss.item()
                if self.dataset_type == "mvtec3d" and self.use_floss:
                    feature_loss = self.contrastive(model_output)
                    # feature_loss = compute_diff_modality_loss(model_output)
                    loss = noise_loss + 0.001 * feature_loss
                    epoch_noise_loss += noise_loss.item()
                    epoch_feature_loss += feature_loss.item()
                else:
                    loss = noise_loss
                 
                loss.backward()
                epoch_loss += loss.item()   
                nn.utils.clip_grad_norm_(self.controllora.parameters(), args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
            
            epoch_loss /= len(self.train_dataloader)
            epoch_noise_loss /= len(self.train_dataloader)
            epoch_feature_loss /= len(self.train_dataloader)

            loss_list.append(epoch_loss)
            noise_loss_list.append(epoch_noise_loss)
            feature_loss_list.append(epoch_feature_loss)
            
            print('Training-Epoch {} Loss: {:.6f}, Noise loss: {:.6f}, Feature loss:{:.6f}'.format(epoch, epoch_loss, epoch_noise_loss, epoch_feature_loss))
            self.train_log_file.write('Training-Epoch {} Loss: {:.6f}, Noise loss: {:.6f}, Feature loss:{:.6f}\n'.format(epoch, epoch_loss, epoch_noise_loss, epoch_feature_loss))

            # save model
            with torch.no_grad():
                if epoch % self.save_epoch == 0:
                    export_loss(args.ckpt_path + '/total_loss.png', loss_list)
                    if self.use_floss:
                        export_loss(args.ckpt_path + '/noise_loss.png', noise_loss_list)
                        export_loss(args.ckpt_path + '/feature_loss.png', feature_loss_list)

                    #self.memorybank_testing.Evaluation(epoch)
                    val_loss = self.log_validation() # Evaluate
                    val_loss_list.append(val_loss)
                    export_loss(args.ckpt_path + '/val_loss.png', val_loss_list)
                    if val_loss < val_best_loss:
                        val_best_loss = val_loss
                        model_path = args.ckpt_path + f'/best_controlnet.pth'
                        torch.save(self.controllora.state_dict(), model_path)
                        print("### Save Model ###")
                        
                if epoch % 25 == 0 and epoch != 0:
                    model_path = args.ckpt_path + f'/epoch{epoch}_controlnet.pth'
                    torch.save(self.controllora.state_dict(), model_path)
                    print("### Save Epoch Model ###")

if __name__ == "__main__":

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device = {device}")

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    runner = TrainUnet(args=args, device=device)
    runner.train()