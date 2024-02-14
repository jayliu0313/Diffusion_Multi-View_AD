import torch
import numpy as np
import os
import os.path as osp
import argparse

from core.data import train_lightings_loader, val_lightings_loader
from core.models.controllora import ControlLoRAModel
from not_use.autoencoder import Autoencoder


from transformers import CLIPTextModel, AutoTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from not_use.backnone import RGB_Extractor
from torchvision import transforms
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training Setup
parser = argparse.ArgumentParser(description='train_resnet')
parser.add_argument('--data_path', type=str, default="/mnt/home_6T/public/jayliu0313/datasets/Eyecandies/")
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--save_epoch', default=3, type=int)
parser.add_argument("--num_train_epochs", type=int, default=1000)
parser.add_argument('--ckpt_path', default="checkpoints/resnet_checkpoints/dino/")
parser.add_argument('--backbone_model', default="vit_base_patch8_224_dino")
parser.add_argument('--learning_rate', default=1e-4)

# Load Checkpoints
parser.add_argument('--load_vae_ckpt_path', type=str, default="./checkpoints/rgb_checkpoints/pretrained_VAE_FCFU/best_ckpt.pth")
parser.add_argument('--load_controlnet_ckpt_path', type=str, default="/mnt/home_6T/public/jayliu0313/check_point/Diffusion_ckpt/controlnet_model_fcInput_nmapCondition/controlnet_epoch_100.pth")
parser.add_argument('--image_size', default=256, type=int)

# Model Setup
#parser.add_argument("--clip_id", type=str, default="openai/clip-vit-base-patch32")
parser.add_argument("--diffusion_id", type=str, default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--revision", type=str, default="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c")
parser.add_argument("--noise_intensity", type=int, default=50)
parser.add_argument("--step_size", type=int, default=2)
parser.add_argument("--controllora_linear_rank", type=int, default=4)
parser.add_argument("--controllora_conv2d_rank", type=int, default=0)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument('--CUDA', type=int, default=0, help="choose the device of CUDA")

def loss_fucntion(a, b):
    # cos_loss = torch.nn.CosineSimilarity().to(device)
    mse_loss = torch.nn.MSELoss().to(device)
    loss = 0
    for item in range(len(a)):
        # loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1)))
        loss += mse_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1))
    return loss

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
    
class TrainResnet():
    def __init__(self, args, device):
        self.device = device
        self.bs = args.batch_size
        self.image_size = args.image_size
        self.num_train_epochs = args.num_train_epochs
        self.save_epoch = args.save_epoch
        self.noise_intensity = args.noise_intensity
        
        self.train_log_file = open(osp.join(args.ckpt_path, "training_log.txt"), "a", 1)
        self.val_log_file = open(osp.join(args.ckpt_path, "val_log.txt"), "a", 1)

        # Load training and validation data
        self.train_dataloader = train_lightings_loader(args)
        self.val_dataloader = val_lightings_loader(args)

        # Create Unet Model
        self.tokenizer = AutoTokenizer.from_pretrained(args.diffusion_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.diffusion_id, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.diffusion_id, subfolder="scheduler")
        self.num_timesteps = int(len(self.noise_scheduler.timesteps) / args.step_size)
        self.noise_scheduler.set_timesteps(self.num_timesteps)
        self.timesteps_list = self.noise_scheduler.timesteps[self.noise_scheduler.timesteps <= self.noise_intensity]
        
        AE = Autoencoder(device)
        AE.load_state_dict(torch.load(args.load_vae_ckpt_path, map_location=self.device)['model'])
        self.vae = AE.vae
        self.decomp = AE.decomp_block
        
        self.unet = UNet2DConditionModel.from_pretrained(
                args.diffusion_id,
                subfolder="unet",
                revision=args.revision)

        # Create ControlNet Model 
        self.controllora: ControlLoRAModel
        self.controllora = ControlLoRAModel.from_unet(self.unet, lora_linear_rank=args.controllora_linear_rank, lora_conv2d_rank=args.controllora_conv2d_rank)
        self.controllora.load_state_dict(torch.load(args.load_controlnet_ckpt_path, map_location=self.device))
        self.controllora.tie_weights(self.unet)

        # Create Feature Extractor
        self.feature_extractor = RGB_Extractor(device, args.backbone_model)
    
        self.vae.to(self.device)
        self.decomp.to(self.device)
        self.unet.to(self.device)
        self.text_encoder.to(self.device)
        self.controllora.to(self.device)
        self.feature_extractor.to(self.device)
        
        self.vae.requires_grad_(False)
        self.decomp.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.controllora.requires_grad_(False)
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
            
        self.vae.eval()
        self.decomp.eval()
        self.unet.eval()
        self.text_encoder.eval()
        self.controllora.eval()
        self.feature_extractor.train()    
        
        self.mse_loss = torch.nn.MSELoss().to(device)
        self.image_transform = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def image2latents(self, x):
        x = x * 2.0 - 1.0
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * 0.18215
        return latents
    
    def latents2image(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image
    
    def get_text_embedding(self, text_prompt, bsz):
        with torch.no_grad():
            tok = self.tokenizer(text_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embedding = self.text_encoder(tok.input_ids.to(self.device).repeat(bsz,1))[0]
        return text_embedding
                
    def forward_process_with_T(self, latents, T):
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        timesteps = torch.tensor([T], device=self.device).repeat(bsz)
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        return noise, timesteps, noisy_latents
               
    def reconstruction(self, lightings, nmaps, encoder_hidden_states):
        '''
        The reconstruction process
        :param y: the target image
        :param x: the input image
        :param seq: the sequence of denoising steps
        :param model: the UNet model
        :param x0_t: the prediction of x0 at time step t
        '''
        with torch.no_grad():
            # Convert images to latent space
            #latents = self.get_vae_latents(lightings)
            latents = self.image2latents(lightings)
            fc_latents = self.decomp.get_fc(latents)
            fu_latents = self.decomp.get_fu(latents)
            bsz = fc_latents.shape[0]

            # Add noise to the latents according to the noise magnitude at each timestep
            _, timesteps, noisy_latents = self.forward_process_with_T(fc_latents, self.timesteps_list[0].item()) 

            
            #encoder_hidden_states = lighting_text_embedding # [bs, 77, 768]
            condition_image = nmaps
            # Denoising Loop
            self.noise_scheduler.set_timesteps(self.num_timesteps, device=self.device)
            
            for t in self.timesteps_list:
                
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
            
                noisy_latents = self.noise_scheduler.step(noise_pred.to(self.device), t.item(), noisy_latents.to(self.device)).prev_sample
            
            rec_latents = self.decomp.fuse_both(noisy_latents, fu_latents)
            rec_lightings = self.latents2image(rec_latents)
        return rec_lightings
    
    def log_validation(self, encoder_hidden_states):
        val_loss = 0.0
        for lightings, nmaps in tqdm(self.val_dataloader, desc="Validation"):

            with torch.no_grad():
                lightings = lightings.to(self.device).view(-1, 3, self.image_size, self.image_size)
                nmaps = nmaps.to(self.device).repeat_interleave(6, dim=0)

                rec_lightings = self.reconstruction(lightings, nmaps, encoder_hidden_states)

                rec_lightings = self.image_transform(rec_lightings)
                reconst_fe = self.feature_extractor(rec_lightings)

                target = self.image_transform(lightings)
                target_fe = self.feature_extractor(target)

                loss = self.mse_loss(reconst_fe, target_fe)
                val_loss += loss.item()

                
        val_loss /= len(self.val_dataloader)
        print('Validation Loss: {:.6f}'.format(val_loss))
        self.val_log_file.write('Validation Loss: {:.6f}\n'.format(val_loss))
        return val_loss
    
    def train(self):
        optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr = args.learning_rate)

        
        loss_list = []
        # Get CLIP embeddings

        encoder_hidden_states = self.get_text_embedding("", self.bs*6) # [bs*6, 77, 768]
        
        for epoch in range(self.num_train_epochs):
            
            epoch_loss = 0.0
            for lightings, nmaps in tqdm(self.train_dataloader, desc="Training"):
               
                lightings = lightings.to(self.device).view(-1, 3, self.image_size, self.image_size)
                nmaps = nmaps.to(self.device).repeat_interleave(6, dim=0)
                
        
                rec_lightings = self.reconstruction(lightings, nmaps, encoder_hidden_states)
                rec_lightings = self.image_transform(rec_lightings)
                reconst_fe = self.feature_extractor(rec_lightings)
              
                target = self.image_transform(lightings)
                target_fe = self.feature_extractor(target)
    
                loss = self.mse_loss(reconst_fe, target_fe)
                print(loss)
                loss.backward()
                optimizer.zero_grad()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            epoch_loss /= len(self.train_dataloader)
            loss_list.append(epoch_loss)
            print('Training - Epoch {}: Loss: {:.6f}'.format(epoch, epoch_loss))
            self.train_log_file.write('Training - Epoch {}: Loss: {:.6f}\n'.format(epoch, epoch_loss))
            #  inference
            if epoch % self.save_epoch == 0:
                export_loss(args.ckpt_path + '/loss.png', loss_list)
                val_loss = self.log_validation(encoder_hidden_states) # Evaluate
                if val_loss < val_best_loss:
                    val_best_loss = val_loss
                    model_path = args.ckpt_path + f'/feature_extractor_{epoch}.pth'
                    torch.save(self.feature_extractor.state_dict(), model_path)
                    print("### Save Model ###")


if __name__ == "__main__":

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device = {device}")

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
        
    runner = TrainResnet(args=args, device=device)
    runner.train()