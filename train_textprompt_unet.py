import os
import os.path as osp
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from core.data import train_lightings_loader, val_lightings_loader
from utils.visualize_util import display_image
import torch.nn.functional as F
from transformers import CLIPTextModel, AutoTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
import matplotlib.pyplot as plt
import matplotlib
import random

matplotlib.use('Agg') 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--data_path', default="/mnt/home_6T/public/jayliu0313/datasets/Eyecandies/", type=str)
parser.add_argument('--ckpt_path', default="checkpoints/diffusion_checkpoints/DiffusionV1-4_TextPromptAnomalyNormal")
parser.add_argument("--load_vae_ckpt", default="")
parser.add_argument("--load_decom_ckpt", default="")
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument("--save_epoch", type=int, default=3)

# Model Setup
#parser.add_argument("--clip_id", type=str, default="openai/clip-vit-base-patch32")
parser.add_argument("--diffusion_id", type=str, default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--revision", type=str, default="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c")
parser.add_argument("--controlnet_id", type=str, default=None)
parser.add_argument(
        "--controllora_linear_rank",
        type=int,
        default=4,
        help=("The dimension of the Linear Module LoRA update matrices."),
    )
parser.add_argument(
        "--controllora_conv2d_rank",
        type=int,
        default=0,
        help=("The dimension of the Conv2d Module LoRA update matrices."),
    )

# Training Setup
parser.add_argument("--anomaly_prob", type=float, default=0.5)
parser.add_argument("--anomaly_scale", type=float, default=0.8)

parser.add_argument("--learning_rate", default=5e-6)
parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
parser.add_argument("--workers", default=6)
parser.add_argument('--CUDA', type=int, default=0, help="choose the device of CUDA")
parser.add_argument("--lr_scheduler", type=str, default="constant", help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'' "constant", "constant_with_warmup"]'),)
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument('--epoch', default=0, type=int, help="Which epoch to start training at")
parser.add_argument("--num_train_epochs", type=int, default=1000)
parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")


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


class TrainTextUnet():
    def __init__(self, args, device):

        self.device = device
        self.bs = args.batch_size
        self.image_size = args.image_size
        self.num_train_epochs = args.num_train_epochs
        self.save_epoch = args.save_epoch
        self.prob = args.anomaly_prob
        self.scale = args.anomaly_scale
        self.train_log_file = open(osp.join(args.ckpt_path, "training_log.txt"), "a", 1)
        self.val_log_file = open(osp.join(args.ckpt_path, "val_log.txt"), "a", 1)

        # Load training and validation data
        self.train_dataloader = train_lightings_loader(args)
        self.val_dataloader = val_lightings_loader(args)

        # Create Model
        self.tokenizer = AutoTokenizer.from_pretrained(args.diffusion_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.diffusion_id, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.diffusion_id, subfolder="scheduler")

        # Load vae model
        self.vae = AutoencoderKL.from_pretrained(
                    args.diffusion_id,
                    subfolder="vae",
                    revision=args.revision,
                    torch_dtype=torch.float32
                )
        # self.decomp_block = Decom_Block(4)
        
        self.unet = UNet2DConditionModel.from_pretrained(
                args.diffusion_id,
                subfolder="unet",
                revision=args.revision)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)

        self.vae.to(self.device)
        self.unet.to(self.device)
        self.text_encoder.to(self.device)

        # Get CLIP embeddings
        with torch.no_grad():
            self.encoder_hidden_states_a = self.get_text_embedding(["anomaly"], self.bs * 6) # [bs * 6, 77, 768]
            self.encoder_hidden_states_n = self.get_text_embedding(["normal"], self.bs * 6) # [bs * 6, 77, 768]

        # Optimizer creation
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
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
        self.defect_nums = 0
        self.normal_nums = 0
        
    def add_jitter(self, feature_tokens):
        B, D, H, W = feature_tokens.shape
        feature_tokens = feature_tokens.reshape(B, D, H*W)
        if random.uniform(0, 1) <= self.prob:
            batch_size, dim_channel, num_tokens = feature_tokens.shape
            feature_norms = (
                feature_tokens.norm(dim=1).unsqueeze(1) / dim_channel
            )  # B x 1 X (H x W)
            jitter = torch.randn((batch_size, dim_channel, num_tokens)).to(feature_tokens.device)
            jitter = jitter * feature_norms * self.scale
            feature_tokens = feature_tokens + jitter
            context_embedding = self.encoder_hidden_states_a
            self.defect_nums += 1
        else:
            context_embedding = self.encoder_hidden_states_n
            self.normal_nums += 1
        return feature_tokens.reshape(B, D, H, W), context_embedding

    def image2latents(self, x):
        with torch.no_grad():
            x = x * 2.0 - 1.0
            latents = self.vae.encode(x).latent_dist.sample()
            latents = latents * 0.18215
            
        return latents
    
    def latents2image(self, latents):
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            image = self.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
        return image
    
    def forward_process(self, x_0):
        noise = torch.randn_like(x_0) # Sample noise that we'll add to the latents
        bsz = x_0.shape[0]
        
        timestep = torch.randint(1, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device) # Sample a random timestep for each image
        timestep = timestep.long()
        x_t = self.noise_scheduler.add_noise(x_0, noise, timestep) # Corrupt image
        return noise, timestep, x_t

    def get_text_embedding(self, text_prompt, bsz):
        tok = self.tokenizer(text_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embedding = self.text_encoder(tok.input_ids.to(self.device).repeat(bsz,1))[0]
        return text_embedding

    def visualize_anomaly(self):
        visualize_path =  os.path.join(args.ckpt_path, "artificial_anomaly")
        if not os.path.exists(visualize_path):
            os.makedirs(visualize_path)
        
        for i, (lightings, _) in enumerate(tqdm(self.train_dataloader, desc="Training")):
            with torch.no_grad():
                lightings = lightings.to(self.device).view(-1, 3, self.image_size, self.image_size) # [bs * 6, 3, 256, 256]
                # nmaps = nmaps.to(self.device).repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]
                
                # Convert images to latent space
                latents = self.image2latents(lightings)
                latents, _ = self.add_jitter(latents)
                rec_images = self.latents2image(latents)
                
                if i % 10 == 0:
                    lightings = lightings.view(-1, 6, 3, self.image_size, self.image_size)
                    rec_images = rec_images.view(-1, 6, 3, self.image_size, self.image_size)
                    lighting = lightings[0, :, :, :]
                    rec_image = rec_images[0, :, :, :]
                    display_image(lighting, rec_image, visualize_path, i)
                
    def log_validation(self):
        val_loss = 0.0
        for lightings, nmaps in tqdm(self.val_dataloader, desc="Validation"):

            with torch.no_grad():
                
                lightings = lightings.to(self.device).view(-1, 3, self.image_size, self.image_size) # [bs * 6, 3, 256, 256]
                # nmaps = nmaps.to(self.device).repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]

                # Convert images to latent space
                latents = self.image2latents(lightings)
                latents, context_embedding = self.add_jitter(latents)
                # Add noise to the latents according to the noise magnitude at each timestep
                noise, timesteps, noisy_latents = self.forward_process(latents)

                # Get CLIP embeddings
                # encoder_hidden_states = self.get_text_embedding(self.encoder_hidden_states, self.bs * 6) # [bs * 6, 77, 768]

                # Training ControlNet
                model_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=context_embedding,
                ).sample

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                val_loss += loss.item()

                
        val_loss /= len(self.val_dataloader)
        print('Validation Loss: {:.6f}'.format(val_loss))
        self.val_log_file.write('Validation Loss: {:.6f}\n'.format(val_loss))
        return val_loss

    def train(self):
        # Start Training #
        loss_list = []
        val_best_loss = float('inf')
        self.visualize_anomaly()
        for epoch in range(self.num_train_epochs):
            self.defect_nums = 0
            self.normal_nums = 0
            epoch_loss = 0.0
            for lightings, _ in tqdm(self.train_dataloader, desc="Training"):

                self.optimizer.zero_grad()
                lightings = lightings.to(self.device).view(-1, 3, self.image_size, self.image_size) # [bs * 6, 3, 256, 256]
                # nmaps = nmaps.to(self.device).repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]
                
                # Convert images to latent space
                latents = self.image2latents(lightings)
                latents, context_embedding = self.add_jitter(latents)
                # Add noise to the latents according to the noise magnitude at each timestep
                noise, timesteps, noisy_latents = self.forward_process(latents) 
                
                # Predict the noise from Unet
                model_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=context_embedding,
                ).sample

                # Compute loss and optimize model parameter
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                loss.backward()
                epoch_loss += loss.item()
                nn.utils.clip_grad_norm_(self.unet.parameters(), args.max_grad_norm)
                
                self.optimizer.step()
                self.lr_scheduler.step()
            print("Defect Numbers:", self.defect_nums)
            print("Normal Numbers:", self.normal_nums)
            epoch_loss /= len(self.train_dataloader)
            loss_list.append(epoch_loss)
            print('Training - Epoch {}: Loss: {:.6f}'.format(epoch, epoch_loss))
            self.train_log_file.write('Training - Epoch {}: Loss: {:.6f}\n'.format(epoch, epoch_loss))

            # save model
            if epoch % self.save_epoch == 0:
                export_loss(args.ckpt_path + '/loss.png', loss_list)
                val_loss = self.log_validation() # Evaluate
                if val_loss < val_best_loss:
                    val_best_loss = val_loss
                    model_path = args.ckpt_path + f'/best_unet_ckpt.pth'
                    torch.save(self.unet.state_dict(), model_path)
                    print("### Save Model ###")


    
if __name__ == "__main__":

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device = {device}")

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
        
    runner = TrainTextUnet(args=args, device=device)
    runner.train()