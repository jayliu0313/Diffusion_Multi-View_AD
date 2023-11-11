import torch
import torch.nn as nn
import argparse
import os
import os.path as osp
import math
import itertools
from tqdm import tqdm

from core.models.network_util import Decom_Block
from core.data import train_lightings_loader, val_lightings_loader

import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL
# from diffusers import StableDiffusionImg2ImgPipeline
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--data_path', default="/mnt/home_6T/public/jayliu0313/datasets/Eyecandies/", type=str)
parser.add_argument('--ckpt_path', default="checkpoints/rgb_checkpoints/train_VAE_stable-diffusion-v1-4_meanfcloss_probchangefcfu")
# parser.add_argument('--diffusion_id', default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--load_vae_ckpt", default=None)
parser.add_argument("--load_decom_ckpt", default=None)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--val_every', default=3, type=int)

# Training Setup
parser.add_argument("--rand_weight", default=0.3, type=float)
parser.add_argument("--training_mode", default="fuse_fc", help="traing type: mean, fuse_fc, fuse_both, random_both")
parser.add_argument("--model", default="Masked_Conv", help="traing type: Conv, Conv_Ins, Masked_Conv")
parser.add_argument("--learning_rate", default=1e-4)
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

class Pipline():
    def __init__(self, weight_dtype=torch.float32, diffusion_id="CompVis/stable-diffusion-v1-4", revision="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c"):
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = weight_dtype
        
        # decomposition model
        self.decomp_block = Decom_Block(4).to(self.device)
        
        # Load vae model
        self.vae = AutoencoderKL.from_pretrained(
                    diffusion_id,
                    subfolder="vae",
                    revision=revision,
                    torch_dtype=weight_dtype
                ).to(self.device)
        self.vae.requires_grad_(False)
        self.vae_trainable_params = []
        for name, param in self.vae.named_parameters():
            if 'decoder' in name:
                param.requires_grad = True
                self.vae_trainable_params.append(param)
                
        self.load_ckpt(args.load_vae_ckpt, args.load_decom_ckpt)
        print("current device:", self.device)
        
    def load_ckpt(self, vae_ckpt, decomp_ckpt):
        if vae_ckpt is not None:
            self.vae.load_state_dict(torch.load(vae_ckpt, map_location=self.device))
            print("load unet ckpt from", vae_ckpt)
        if decomp_ckpt is not None:
            self.decomp_block.load_state_dict(torch.load(decomp_ckpt, map_location=self.device))
            print("load adapter ckpt from", decomp_ckpt)
            
    def image2latents(self, x):
        x = x.float() * 2.0 - 1.0
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * 0.18215
        return latents
    
    def latents2image(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        return image
    
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
            itertools.chain(self.decomp_block.parameters(), self.vae_trainable_params)
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
        loss_list = []
        best_val_loss = float('inf')
        
        for epoch in range(args.epoch, args.num_train_epochs):
            self.vae.train()
            self.decomp_block.train()
            epoch_loss = 0.0
            epoch_rec_loss = 0.0
            epoch_fc_loss = 0.0
            # i = 0
            for lightings, _ in tqdm(train_dataloader):
                # if i == 5:
                #     break
                # i+=1
                lightings = lightings.view(-1, 3, args.image_size, args.image_size).to(self.device, dtype=self.dtype)

                latents = self.image2latents(lightings)
                latents, loss_fc = self.decomp_block.prob_rand_forward_meanfcloss(latents)
                out = self.latents2image(latents)
                rec_loss = F.mse_loss(out.float(), (lightings*2.0-1.0).float(), reduction="mean")
                loss = rec_loss + loss_fc * 0.1
                
                epoch_loss += loss.item()
                epoch_rec_loss += rec_loss.item()
                epoch_fc_loss += loss_fc.item()
                loss.backward()
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()                
                
            epoch_loss /= len(train_dataloader)
            epoch_rec_loss /= len(train_dataloader)
            epoch_fc_loss /= len(train_dataloader)
            loss_list.append(epoch_loss)
            print('Training - Epoch {}: Loss: {:.6f}, Rec Loss: {:.6f}, FC Loss:{:.6f}'.format(epoch, epoch_loss, epoch_rec_loss, epoch_fc_loss))
            train_log_file.write('Training - Epoch {}: Loss: {:.6f}, Rec Loss: {:.6f}, FC Loss:{:.6f}\n'.format(epoch, epoch_loss, epoch_rec_loss, epoch_fc_loss))
            
            # evaluation
            if epoch % args.val_every == 0 or epoch == args.num_train_epochs - 1:
                self.vae.eval()
                self.decomp_block.eval()
                epoch_val_loss = 0.0
                epoch_val_rec_loss = 0.0
                epoch_val_fc_loss = 0.0
                # i = 0
                for lightings, _ in tqdm(val_dataloader):
                    with torch.no_grad():
                        # if i == 5:
                        #     break
                        # i+=1
                        lightings = lightings.view(-1, 3, args.image_size, args.image_size).to(self.device, dtype=self.dtype)
                        latents = self.image2latents(lightings)
                        latents, loss_fc = self.decomp_block.fusefc_prob_forward(latents)
                        out = self.latents2image(latents)
                        rec_loss = F.mse_loss(out.float(), (lightings*2.0-1.0).float(), reduction="mean")
                        loss = rec_loss + loss_fc * 0.1
                        
                        epoch_val_loss += loss.item()
                        epoch_val_rec_loss += rec_loss.item()
                        epoch_val_fc_loss += loss_fc.item()    
                        
                epoch_val_loss /= len(val_dataloader)
                epoch_val_rec_loss /= len(val_dataloader) 
                epoch_val_fc_loss /= len(val_dataloader)             
                print('Validation - Epoch {}: Loss: {:.6f}, Rec Loss: {:.6f}, FC Loss:{:.6f}'.format(epoch, epoch_val_loss, epoch_val_rec_loss, epoch_val_fc_loss))
                val_log_file.write('Validation - Epoch {}: Loss: {:.6f}, Rec Loss: {:.6f}, FC Loss:{:.6f}\n'.format(epoch, epoch_val_loss, epoch_val_rec_loss, epoch_val_fc_loss))
                export_loss(args.ckpt_path + '/training_loss.png', loss_list)
                
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    model_path = args.ckpt_path+f'/best_vae_ckpt.pth'
                    torch.save(self.vae.state_dict(), model_path)
                    model_path = args.ckpt_path+f'/best_decomp_ckpt.pth'
                    torch.save(self.decomp_block.state_dict(), model_path)
                    print("Save the best checkpoint")
        
        val_log_file.close()
        train_log_file.close()

    
if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
        
    Training = Pipline()
    Training.run(args)