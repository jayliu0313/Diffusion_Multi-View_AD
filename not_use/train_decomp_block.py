import torch
import torch.nn as nn
import argparse
import os
import os.path as osp
import math
import itertools
from tqdm import tqdm

from not_use.backnone import RGB_Extractor
from core.models.network_util import Decom_Block
from core.data import train_lightings_loader, val_lightings_loader
from torchvision.transforms import transforms

import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--data_path', default="/mnt/home_6T/public/jayliu0313/datasets/Eyecandies/", type=str)
parser.add_argument('--ckpt_path', default="checkpoints/rgb_checkpoints/DinoLock_DeocompBlock_MeanFC_RandFU_MSELoss")
parser.add_argument("--backbone_name", default="vit_base_patch8_224_dino")
parser.add_argument("--out_indices", default=None)
parser.add_argument("--load_backbone_ckpt", default=None)
parser.add_argument("--load_decomp_ckpt", default=None)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--save_epoch', default=3, type=int)

# Training Setup
parser.add_argument("--rand_weight", default=0.3, type=float)
parser.add_argument("--training_mode", default="fuse_fc", help="traing type: mean, fuse_fc, fuse_both, random_both")
parser.add_argument("--model", default="Masked_Conv", help="traing type: Conv, Conv_Ins, Masked_Conv")
parser.add_argument("--learning_rate", default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
parser.add_argument("--workers", default=8)
parser.add_argument('--CUDA', type=int, default=0, help="choose the device of CUDA")
parser.add_argument("--lr_scheduler", type=str, default="constant", help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'' "constant", "constant_with_warmup"]'),)
parser.add_argument('--epoch', default=0, type=int, help="Which epoch to start training at")
parser.add_argument("--num_train_epochs", type=int, default=1000)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# def loss_fucntion(a, b,):
#     cos_loss = torch.nn.CosineSimilarity().to(device)
#     # mse_loss = torch.nn.MSELoss().to(device)

#     loss = torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1)))
#     # loss = loss_function(a.view(a.shape[0], -1), b.view(b.shape[0], -1))
#     return loss

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

class TrainDecompoistionBlock():
    def __init__(self, args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.bs = args.batch_size
        self.image_size = args.image_size
        self.num_train_epochs = args.num_train_epochs
        self.save_epoch = args.save_epoch
        self.train_log_file = open(osp.join(args.ckpt_path, "training_log.txt"), "a", 1)
        self.val_log_file = open(osp.join(args.ckpt_path, "val_log.txt"), "a", 1)

        # Load training and validation data
        self.train_dataloader = train_lightings_loader(args)
        self.val_dataloader = val_lightings_loader(args)

        # Load vae model
        self.feature_extractor = RGB_Extractor(device=device, backbone_name=args.backbone_name)
        self.decomp_block = Decom_Block(768)
        # Load VAE checkpoint  
        if args.load_backbone_ckpt is not None:
            print("Load best vae checkpoints")
            self.feature_extractor.load_state_dict(torch.load(args.load_backbone_ckpt, map_location=self.device))
        if args.load_decomp_ckpt is not None:
            print("Load best decomposition block checkpoints")
            self.decomp_block.load_state_dict(torch.load(args.load_decomp_ckpt, map_location=self.device))


        self.feature_extractor.requires_grad_(False)
        self.decomp_block.requires_grad_(True)

        self.feature_extractor.to(self.device)
        self.decomp_block.to(self.device)
        
        # params_to_optimize = (
        #     itertools.chain(self.feature_extractor.parameters(), self.decomp_block.parameters())
        # )
        params_to_optimize = (
            itertools.chain(self.decomp_block.parameters())
        )
        # Optimizer creation
        self.optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=args.learning_rate,
        )
        
        self.mse_loss = torch.nn.MSELoss().to(device)
        self.cos_loss = torch.nn.CosineSimilarity().to(device)
        self.image_transform = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
         
    def log_validation(self, text_embedding):
        val_loss = 0.0
        for lightings, nmaps in tqdm(self.val_dataloader, desc="Validation"):

            with torch.no_grad():
                
                lightings = lightings.to(self.device).view(-1, 3, self.image_size, self.image_size) # [bs * 6, 3, 256, 256]
                # nmaps = nmaps.to(self.device).repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]
                features = self.feature_extractor(self.image_transform(lightings))
                rand_features = self.decomp_block.meanfc_prbo_randfu_forward(features)
                # loss = torch.mean(1-self.cos_loss(features.reshape(features.shape[0], -1), rand_features.reshape(rand_features.shape[0], -1)))

                loss = self.mse_loss(features, rand_features)
                val_loss += loss.item()

                
        val_loss /= len(self.val_dataloader)
        print('Validation Loss: {:.6f}'.format(val_loss))
        self.val_log_file.write('Validation Loss: {:.6f}\n'.format(val_loss))
        return val_loss

    def train(self):
        text_prompt = ""

        # Start Training #
        loss_list = []
        val_best_loss = float('inf')
        for epoch in range(self.num_train_epochs):

            epoch_loss = 0.0
            for lightings, nmaps in tqdm(self.train_dataloader, desc="Training"):

                self.optimizer.zero_grad()
                lightings = lightings.to(self.device).view(-1, 3, self.image_size, self.image_size) # [bs * 6, 3, 256, 256]
                # nmaps = nmaps.to(self.device).repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]
                # with torch.no_grad():
                features = self.feature_extractor(self.image_transform(lightings))
                rand_features = self.decomp_block.meanfc_prbo_randfu_forward(features)
                loss = self.mse_loss(features, rand_features)
                # loss = torch.mean(1-self.cos_loss(features.reshape(features.shape[0], -1), rand_features.reshape(rand_features.shape[0], -1)))
                # print(loss)
                # Compute loss and optimize model parameter
                loss.backward()
                epoch_loss += loss.item()                
                self.optimizer.step()
                
            epoch_loss /= len(self.train_dataloader)
            loss_list.append(epoch_loss)
            print('Training - Epoch {}: Loss: {:.6f}'.format(epoch, epoch_loss))
            self.train_log_file.write('Training - Epoch {}: Loss: {:.6f}\n'.format(epoch, epoch_loss))

            # save model
            if epoch % self.save_epoch == 0:
                export_loss(args.ckpt_path + '/loss.png', loss_list)
                val_loss = self.log_validation(text_embedding=text_prompt) # Evaluate
                if val_loss < val_best_loss:
                    val_best_loss = val_loss
                    model_path = args.ckpt_path + f'/decomposition_best_ckpt.pth'
                    state_dict = {
                        'backbone': self.feature_extractor.state_dict(),
                        'decomp_block': self.decomp_block.state_dict(),
                        'current_iteration': epoch,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }
                    torch.save(state_dict, model_path)
                    print("### Save Model ###")
    
if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
        
    Training = TrainDecompoistionBlock(args)
    Training.train()