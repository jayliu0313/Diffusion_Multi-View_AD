import os
import os.path as osp
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from core.data import *

import torch.nn.functional as F
from transformers import CLIPTextModel, AutoTokenizer
from diffusers import AutoencoderKL, DDPMScheduler
from core.models.unet_model import build_unet
from utils.utils import t2np
from diffusers.optimization import get_scheduler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--data_path', default="/mnt/home_6T/public/samchu0218/Raw_Datasets/MVTec_AD/MVTec_Loco/", type=str)
# "/mnt/home_6T/public/jayliu0313/datasets/mvtec3d_preprocessing/"
# "/mnt/home_6T/public/samchu0218/Raw_Datasets/MVTec_AD/MVTec_Loco/"

parser.add_argument('--dataset_type', default="eyecandies", help="eyecandies, mvtec3d, mvtecloco")
parser.add_argument('--ckpt_path', default="checkpoints/diffusion_checkpoints/TrainMVTecLoco_RGBEdgemap")
parser.add_argument('--load_unet_ckpt', default="")
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--batch_size', default=2, type=int)

# Model Setup
#parser.add_argument("--clip_id", type=str, default="openai/clip-vit-base-patch32")
parser.add_argument("--diffusion_id", type=str, default="CompVis/stable-diffusion-v1-4", help="CompVis/stable-diffusion-v1-4, runwayml/stable-diffusion-v1-5")

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

def denormalization(x):
    x = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)
    return x

def cos_loss(a, b):
    cos_loss = nn.CosineSimilarity()
    loss = torch.mean(1-cos_loss(a.view(a.shape[0],-1),b.view(b.shape[0],-1)))
    return loss

def display(image, save_path):
    # 將張量轉換為numpy數組
    image_array = image.permute(1, 2, 0).numpy()

    # 顯示圖像
    plt.imshow(image_array)
    plt.axis('off')  # 關閉坐標軸
    plt.savefig(save_path)

class TrainUnet():
    def __init__(self, args, device):

        self.device = device
        self.bs = args.batch_size
        self.image_size = args.image_size
        self.num_train_epochs = args.num_train_epochs
        self.save_epoch = args.save_epoch
        self.viz_save_path = osp.join(args.ckpt_path, "visualize")
        self.train_log_file = open(osp.join(args.ckpt_path, "training_log.txt"), "a", 1)
        self.val_log_file = open(osp.join(args.ckpt_path, "val_log.txt"), "a", 1)

        if not os.path.exists(self.viz_save_path):
            os.makedirs(self.viz_save_path)

        # Load training and validation data
        if args.dataset_type == "eyecandies":
            self.train_dataloader = train_lightings_loader(args)
            self.val_dataloader = val_lightings_loader(args)
        elif args.dataset_type == "mvtec3d":
            self.train_dataloader = mvtec3D_train_loader(args)
            self.val_dataloader = mvtec3D_val_loader(args)
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
        ).to(self.device)
        # self.adapter = nn.Sequential(
        #     nn.Conv2d(4, 4, 3, padding=1),
        # )

        self.unet = build_unet(args)
        if os.path.isfile(args.load_unet_ckpt):
            print("Success load unet checkpoints!")
            self.unet.load_state_dict(torch.load(args.load_unet_ckpt, map_location=self.device))

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
        # self.encoder_hidden_states = self.get_text_embedding("", self.bs * 6)
        self.data_type = args.dataset_type
        
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
        epoch_nmap_noise_loss = 0.0
        epoch_rgb_noise_loss = 0.0
        epoch_feature_loss = 0.0
        epoch_cos_loss = 0.0
        i = 0
        for lightings, nmaps, text_prompt in tqdm(self.val_dataloader, desc="Validation"):
            # i+=1
            # if i == 5:
            #     break
            with torch.no_grad():
                # print(text_prompt)
                lightings = lightings.to(self.device).view(-1, 3, self.image_size, self.image_size) # [bs * 6, 3, 256, 256]
                text_embedding = self.get_text_embedding(text_prompt)
                text_embeddings = text_embedding
                # Convert images to latent space
                img_latents = self.image2latents(lightings)
                # Add noise to the latents according to the noise magnitude at each timestep
                noise, timesteps, noisy_latents = self.forward_process(img_latents)

                if self.data_type == "eyecandies":
                    text_embeddings = text_embedding.repeat_interleave(6, dim=0)

                # Get CLIP embeddings
                 # [bs * 6, 77, 768]
                # Predict the noise from Unet

                model_output = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                )

                pred_noise = model_output['sample']
                noise_loss = F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")

                nmap_latents = self.image2latents(nmaps.to(self.device))
                nmap_noise, nmaps_timesteps, nmap_noisy_latents = self.forward_process(nmap_latents)
                nmap_model_output = self.unet(
                    nmap_noisy_latents,
                    nmaps_timesteps,
                    encoder_hidden_states=text_embedding,
                )
                nmap_pred_noise = nmap_model_output['sample']
                nmap_noise_loss = F.mse_loss(nmap_pred_noise.float(), nmap_noise.float(), reduction="mean")

                loss = 0.2 * nmap_noise_loss + 0.8 * noise_loss #+ Lambda * feature_loss
                val_loss += loss.item()
                epoch_nmap_noise_loss += nmap_noise_loss.item()
                epoch_rgb_noise_loss += noise_loss.item()
                # epoch_feature_loss += feature_loss.item()

        val_loss /= len(self.val_dataloader)
        epoch_nmap_noise_loss /= len(self.val_dataloader)
        epoch_rgb_noise_loss /= len(self.val_dataloader)
        epoch_feature_loss /= len(self.val_dataloader)
        print('Validation Loss: {:.6f}, rgb noise loss: {:.6f}, nmap noise loss: {:.6f}, feature loss:{:.6f}'.format(val_loss, epoch_rgb_noise_loss, epoch_nmap_noise_loss, epoch_feature_loss))
        self.val_log_file.write('Validation Loss: {:.6f}, rgb noise loss: {:.6f}, nmap noise loss: {:.6f}, feature loss:{:.6f}\n'.format(val_loss, epoch_rgb_noise_loss, epoch_nmap_noise_loss, epoch_feature_loss))
        return val_loss

    def train(self):
        # Start Training #
        loss_list = []
        rgb_noise_loss_list = []
        nmap_noise_loss_list = []
        feature_loss_list = []
        val_best_loss = float('inf')
        for epoch in range(self.num_train_epochs):

            epoch_loss = 0.0
            epoch_nmap_noise_loss = 0.0
            epoch_rgb_noise_loss = 0.0
            epoch_feature_loss = 0.0
            epoch_cos_loss = 0.0
            i = 0
            for images, nmaps, text_prompt in tqdm(self.train_dataloader, desc="Training"):
                # print(lightings.shape)
                # self.visualize(lightings[0], nmaps[0], i)
                if i % 40 == 0 and epoch == 0:
                    display(nmaps[0], osp.join(self.viz_save_path, "edgemap_" + str(epoch) + "_" + str(i) + ".png"))
                    display(images[0], osp.join(self.viz_save_path, "rgb_" + str(epoch) + "_" + str(i) + ".png"))
                i+=1
                self.optimizer.zero_grad()
                lightings = images.to(self.device).view(-1, 3, self.image_size, self.image_size) # [bs * 6, 3, 256, 256]

                text_embedding = self.get_text_embedding(text_prompt)
                text_embeddings = text_embedding
                # Convert images to latent space
                latents = self.image2latents(lightings)
                # Add noise to the latents according to the noise magnitude at each timestep
                noise, timesteps, noisy_latents = self.forward_process(latents)
                if self.data_type == "eyecandies":
                    text_embeddings = text_embedding.repeat_interleave(6, dim=0)

                # Predict the noise from Unet
                model_output = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                )

                pred_noise = model_output['sample']

                # Compute loss and optimize model parameter
                noise_loss = F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")

                nmap_latents = self.image2latents(nmaps.to(self.device))
                nmap_noise, nmaps_timesteps, nmap_noisy_latents = self.forward_process(nmap_latents)
                nmap_model_output = self.unet(
                    nmap_noisy_latents,
                    nmaps_timesteps,
                    encoder_hidden_states=text_embedding,
                )

                nmap_pred_noise = nmap_model_output['sample']
                nmap_noise_loss = F.mse_loss(nmap_pred_noise.float(), nmap_noise.float(), reduction="mean")

                loss = nmap_noise_loss + noise_loss
                loss.backward()
                epoch_loss += loss.item()
                epoch_nmap_noise_loss += nmap_noise_loss.item()
                epoch_rgb_noise_loss += noise_loss.item()
                # epoch_feature_loss += feature_loss.item()


                nn.utils.clip_grad_norm_(self.unet.parameters(), args.max_grad_norm)

                self.optimizer.step()
                self.lr_scheduler.step()

            epoch_loss /= len(self.train_dataloader)
            epoch_nmap_noise_loss /= len(self.train_dataloader)
            epoch_rgb_noise_loss /= len(self.train_dataloader)
            epoch_feature_loss /= len(self.train_dataloader)
            epoch_cos_loss /= len(self.train_dataloader)

            loss_list.append(epoch_loss)
            rgb_noise_loss_list.append(epoch_rgb_noise_loss)
            nmap_noise_loss_list.append(epoch_nmap_noise_loss)
            feature_loss_list.append(epoch_feature_loss)

            print('Training - Epoch {} Loss: {:.6f}, rgb noise loss: {:.6f}, nmap noise loss: {:.6f}, feature loss:{:.6f}'.format(epoch, epoch_loss, epoch_rgb_noise_loss, epoch_nmap_noise_loss, epoch_feature_loss))
            self.train_log_file.write('Training - Epoch {} Loss: {:.6f}, rgb noise loss: {:.6f}, nmap noise loss: {:.6f}, feature loss:{:.6f}\n'.format(epoch, epoch_loss, epoch_rgb_noise_loss, epoch_nmap_noise_loss, epoch_feature_loss))

            # save model
            if epoch % 10 == 0 and epoch >= 10:
                model_path = args.ckpt_path + f'/epoch{epoch}_unet.pth'
                torch.save(self.unet.state_dict(), model_path)
                print("### Save Model ###")
            # if epoch == self.num_train_epochs - 1:
            #     model_path = args.ckpt_path + f'/last_unet.pth'
            #     torch.save(self.unet.state_dict(), model_path)
            #     print("### Save Model ###")               
            if epoch % self.save_epoch == 0:
                export_loss(args.ckpt_path + '/total_loss.png', loss_list)
                export_loss(args.ckpt_path + '/rgb_noise_loss.png', rgb_noise_loss_list)
                export_loss(args.ckpt_path + '/nmap_noise_loss.png', nmap_noise_loss_list)
                # export_loss(args.ckpt_path + '/feature_loss.png', feature_loss_list)
                val_loss = self.log_validation() # Evaluate
                if val_loss < val_best_loss:
                    val_best_loss = val_loss
                    model_path = args.ckpt_path + f'/best_unet.pth'
                    torch.save(self.unet.state_dict(), model_path)
                    print("### Save Model ###")

    def visualize(self, img, nmap, item):
        save_path = os.path.join("visualize", str(item) + ".png")

        fig = plt.figure(figsize=(12, 5))
        img = denormalization(t2np(img))
        fig.add_subplot(2, 1, 1)
        plt.imshow(img, cmap='gray')
        plt.axis("off")
        plt.title('Testing Image', fontsize = 15)

        target = denormalization(t2np(nmap))
        fig.add_subplot(2, 1, 2)
        plt.imshow(target, cmap='gray')
        plt.axis("off")
        plt.title('Reconstruct Testing Image', fontsize = 15)

        plt.savefig(save_path, dpi=300)
        plt.close()

if __name__ == "__main__":

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device = {device}")

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    runner = TrainUnet(args=args, device=device)
    runner.train()