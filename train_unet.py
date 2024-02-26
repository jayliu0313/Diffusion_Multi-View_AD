import os
import os.path as osp
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from core.data import train_lightings_loader, val_lightings_loader, mvtec3D_train_loader, mvtec3D_val_loader

import torch.nn.functional as F
from transformers import CLIPTextModel, AutoTokenizer
from diffusers import AutoencoderKL, DDPMScheduler
from core.models.unet_model import MyUNet2DConditionModel

from diffusers.optimization import get_scheduler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--data_path', default="/mnt/home_6T/public/jayliu0313/datasets/mvtec3d_preprocessing/", type=str)
# /mnt/home_6T/public/jayliu0313/datasets/Eyecandies/
# /mnt/home_6T/public/jayliu0313/datasets/mvtec3d_preprocessing/

parser.add_argument('--ckpt_path', default="checkpoints/diffusion_checkpoints/TrainMVTec3DAD_UnetV2-1")
parser.add_argument('--load_vae_ckpt', default=None)
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--batch_size', default=6, type=int)
parser.add_argument('--train_type', default="mvtect3d", help="eyecandies_rgb, eyecandies_nmap, mvtect3d")

# Model Setup
#parser.add_argument("--clip_id", type=str, default="openai/clip-vit-base-patch32")
parser.add_argument("--diffusion_id", type=str, default="stabilityai/stable-diffusion-2-1")
parser.add_argument("--revision", type=str, default="")
# ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c
# Training Setup
parser.add_argument("--learning_rate", default=5e-6)
parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
parser.add_argument("--workers", default=4)
parser.add_argument('--CUDA', type=int, default=0, help="choose the device of CUDA")
parser.add_argument("--lr_scheduler", type=str, default="constant", help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'' "constant", "constant_with_warmup"]'),)
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument('--epoch', default=0, type=int, help="Which epoch to start training at")
parser.add_argument("--num_train_epochs", type=int, default=1000)
parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
parser.add_argument("--save_epoch", type=int, default=2)


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


class TrainUnet():
    def __init__(self, args, device):

        self.device = device
        self.bs = args.batch_size
        self.image_size = args.image_size
        self.num_train_epochs = args.num_train_epochs
        self.save_epoch = args.save_epoch
        self.train_log_file = open(osp.join(args.ckpt_path, "training_log.txt"), "a", 1)
        self.val_log_file = open(osp.join(args.ckpt_path, "val_log.txt"), "a", 1)
        self.type = args.train_type
        # Load training and validation data
        if "eyecandies" in args.train_type:
            self.train_dataloader = train_lightings_loader(args)
            self.val_dataloader = val_lightings_loader(args)
        elif args.train_type == "mvtect3d":
            self.train_dataloader = mvtec3D_train_loader(args)
            self.val_dataloader = mvtec3D_val_loader(args)

        # Create Model
        self.tokenizer = AutoTokenizer.from_pretrained(args.diffusion_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.diffusion_id, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.diffusion_id, subfolder="scheduler")

        self.vae = AutoencoderKL.from_pretrained(
            args.diffusion_id,
            subfolder="vae",
            # revision=args.revision,
        ).to(self.device)


        self.unet = MyUNet2DConditionModel.from_pretrained(
                args.diffusion_id,
                subfolder="unet",
                # revision=args.revision
                )

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(True)
        self.text_encoder.requires_grad_(False)

        self.vae.to(self.device)
        self.unet.to(self.device)
        self.text_encoder.to(self.device)

        # Optimizer creation
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay
        )

        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_dataloader) * args.num_train_epochs,
        )
        # self.encoder_hidden_states = self.get_text_embedding("", self.bs * 6)

    def image2latents(self, x):
        with torch.no_grad():
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

    def get_text_embedding(self, text_prompt, n):
        with torch.no_grad():
            tok = self.tokenizer(text_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embedding = self.text_encoder(tok.input_ids.to(self.device))[0]
            text_embeddings = text_embedding.repeat_interleave(n, dim=0)
        return text_embeddings

    def log_validation(self):
        val_loss = 0.0
        i = 0
        for images, nmaps, text_prompt in tqdm(self.val_dataloader, desc="Validation"):
            # i+=1
            # if i == 5:
            #     break
            with torch.no_grad():
                # print(text_prompt)
                if self.type == "eyecandies_rgb":
                    inputs = images.to(self.device).view(-1, 3, self.image_size, self.image_size) # [bs * 6, 3, 256, 256]
                    text_embeddings = self.get_text_embedding(text_prompt, 6)
                elif self.type == "eyecandies_nmap":
                    inputs = nmaps.to(self.device).view(-1, 3, self.image_size, self.image_size)
                    text_embeddings = self.get_text_embedding(text_prompt, 1)
                else:
                    inputs = images.to(self.device).view(-1, 3, self.image_size, self.image_size)
                    text_embeddings = self.get_text_embedding(text_prompt, 1)
                # Convert images to latent space
                latents = self.image2latents(inputs)

                # Add noise to the latents according to the noise magnitude at each timestep
                noise, timesteps, noisy_latents = self.forward_process(latents)

                # Get CLIP embeddings
                 # [bs * 6, 77, 768]
                # Training ControlNet
                
                

                # Predict the noise from Unet
                model_output = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                )
                pred_noise = model_output['sample']
                if self.type == "eyecandies_rgb":
                    unet_f_layer0 = model_output['up_ft'][0]
                    _, C, H, W = unet_f_layer0.shape
                    mean_unet_f_layer0 = torch.mean(unet_f_layer0.view(-1, 6, C, H, W), dim=1)
                    mean_unet_f_layer0 = mean_unet_f_layer0.repeat_interleave(6, dim=0)

                    unet_f_layer1 = model_output['up_ft'][1]
                    _, C, H, W = unet_f_layer1.shape
                    mean_unet_f_layer1 = torch.mean(unet_f_layer1.view(-1, 6, C, H, W), dim=1)
                    mean_unet_f_layer1 = mean_unet_f_layer1.repeat_interleave(6, dim=0)

                    unet_f_layer2 = model_output['up_ft'][2]
                    _, C, H, W = unet_f_layer2.shape
                    mean_unet_f_layer2 = torch.mean(unet_f_layer2.view(-1, 6, C, H, W), dim=1)
                    mean_unet_f_layer2 = mean_unet_f_layer2.repeat_interleave(6, dim=0)

                    unet_f_layer3 = model_output['up_ft'][3]
                    _, C, H, W = unet_f_layer3.shape
                    mean_unet_f_layer3 = torch.mean(unet_f_layer3.view(-1, 6, C, H, W), dim=1)
                    mean_unet_f_layer3 = mean_unet_f_layer3.repeat_interleave(6, dim=0)

                    # Compute loss and optimize model parameter
                    feature_loss = F.l1_loss(mean_unet_f_layer0, unet_f_layer0, reduction="mean")
                    feature_loss += F.l1_loss(mean_unet_f_layer1, unet_f_layer1, reduction="mean")
                    feature_loss += F.l1_loss(mean_unet_f_layer2, unet_f_layer2, reduction="mean")
                    feature_loss += F.l1_loss(mean_unet_f_layer3, unet_f_layer3, reduction="mean")
                    Lambda = 0.01
                    # Compute loss and optimize model parameter
                    noise_loss = F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")
                    loss = noise_loss + Lambda * feature_loss
                else:
                    noise_loss = F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")
                    loss = noise_loss
                    
                val_loss += loss.item()

        val_loss /= len(self.val_dataloader)
        print('Validation Loss: {:.6f}'.format(val_loss))
        self.val_log_file.write('Validation Loss: {:.6f}\n'.format(val_loss))
        return val_loss

    def train(self):
        # Start Training #
        loss_list = []
        val_best_loss = float('inf')
        for epoch in range(self.num_train_epochs):

            epoch_loss = 0.0
            i = 0
            for images, nmaps, text_prompt in tqdm(self.train_dataloader, desc="Training"):
                # i+=1
                # if i == 5:
                #     break
                # print(text_prompt)

                if self.type == "eyecandies_rgb":
                    inputs = images.to(self.device).view(-1, 3, self.image_size, self.image_size) # [bs * 6, 3, 256, 256]
                    text_embeddings = self.get_text_embedding(text_prompt, 6)
                elif self.type == "eyecandies_nmap":
                    inputs = nmaps.to(self.device).view(-1, 3, self.image_size, self.image_size)
                    text_embeddings = self.get_text_embedding(text_prompt, 1)
                else:
                    inputs = images.to(self.device).view(-1, 3, self.image_size, self.image_size)
                    text_embeddings = self.get_text_embedding(text_prompt, 1)
                # Convert images to latent space
                latents = self.image2latents(inputs)
                # Add noise to the latents according to the noise magnitude at each timestep
                noise, timesteps, noisy_latents = self.forward_process(latents)
                # Predict the noise from Unet
                model_output = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                )

                pred_noise = model_output['sample']
                if self.type == "eyecandies_rgb":
                    unet_f_layer0 = model_output['up_ft'][0]
                    _, C, H, W = unet_f_layer0.shape
                    mean_unet_f_layer0 = torch.mean(unet_f_layer0.view(-1, 6, C, H, W), dim=1)
                    mean_unet_f_layer0 = mean_unet_f_layer0.repeat_interleave(6, dim=0)

                    unet_f_layer1 = model_output['up_ft'][1]
                    _, C, H, W = unet_f_layer1.shape
                    mean_unet_f_layer1 = torch.mean(unet_f_layer1.view(-1, 6, C, H, W), dim=1)
                    mean_unet_f_layer1 = mean_unet_f_layer1.repeat_interleave(6, dim=0)

                    unet_f_layer2 = model_output['up_ft'][2]
                    _, C, H, W = unet_f_layer2.shape
                    mean_unet_f_layer2 = torch.mean(unet_f_layer2.view(-1, 6, C, H, W), dim=1)
                    mean_unet_f_layer2 = mean_unet_f_layer2.repeat_interleave(6, dim=0)

                    unet_f_layer3 = model_output['up_ft'][3]
                    _, C, H, W = unet_f_layer3.shape
                    mean_unet_f_layer3 = torch.mean(unet_f_layer3.view(-1, 6, C, H, W), dim=1)
                    mean_unet_f_layer3 = mean_unet_f_layer3.repeat_interleave(6, dim=0)

                    # Compute loss and optimize model parameter
                    feature_loss = F.l1_loss(mean_unet_f_layer0, unet_f_layer0, reduction="mean")
                    feature_loss += F.l1_loss(mean_unet_f_layer1, unet_f_layer1, reduction="mean")
                    feature_loss += F.l1_loss(mean_unet_f_layer2, unet_f_layer2, reduction="mean")
                    feature_loss += F.l1_loss(mean_unet_f_layer3, unet_f_layer3, reduction="mean")

                    Lambda = 0.01
                    # Compute loss and optimize model parameter
                    noise_loss = F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")
                    loss = noise_loss + Lambda * feature_loss
                else:
                    noise_loss = F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")
                    loss = noise_loss

                loss.backward()
                epoch_loss += loss.item()
                nn.utils.clip_grad_norm_(self.unet.parameters(), args.max_grad_norm)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            epoch_loss /= len(self.train_dataloader)
            loss_list.append(epoch_loss)
            print('Training - Epoch {}: Loss: {:.6f}'.format(epoch, epoch_loss))
            self.train_log_file.write('Training - Epoch {}: Loss: {:.6f}\n'.format(epoch, epoch_loss))
            val_loss = epoch_loss
            # save model
            if epoch % self.save_epoch == 0:
                export_loss(args.ckpt_path + '/loss.png', loss_list)
                _ = self.log_validation() # Evaluate
                if val_loss < val_best_loss:
                    val_best_loss = val_loss
                    model_path = args.ckpt_path + f'/best_unet.pth'
                    torch.save(self.unet.state_dict(), model_path)
                    print("### Save Model ###")



if __name__ == "__main__":

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device = {device}")

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    runner = TrainUnet(args=args, device=device)
    runner.train()