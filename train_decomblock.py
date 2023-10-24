import torch
import argparse
import os
import os.path as osp
import torch.nn.functional as F
from tqdm import tqdm
from core.data import train_lightings_loader, val_lightings_loader
# from core.models.contrastive import Contrastive
from core.models.rgb_network import *
# from core.models.unet_model import UNet_Decom, ResUNet_Decom_AE
from core.models.autoencoder import Autoencoder

# import kornia

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--data_path', default="/mnt/home_6T/public/jayliu0313/datasets/Eyecandies/", type=str)
parser.add_argument('--ckpt_path', default="checkpoints/rgb_checkpoints/autoencoder")
parser.add_argument("--load_ckpt", default=None)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--image_size', default=256, type=int)

# Training Setup
parser.add_argument("--rand_weight", default=0.3, type=float)
parser.add_argument("--training_mode", default="fuse_fc", help="traing type: mean, fuse_fc, fuse_both, random_both")
parser.add_argument("--model", default="Masked_Conv", help="traing type: Conv, Conv_Ins, Masked_Conv")
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
parser.add_argument("--workers", default=8)
parser.add_argument("--epochs", default=1000)
parser.add_argument('--CUDA', type=int, default=0, help="choose the device of CUDA")

# Contrstive Learning
# parser.add_argument("--contrastive_w", default=0.001)
# parser.add_argument("--temperature_f", default=0.5)
# parser.add_argument("--temperature_l", default=1.0)

args = parser.parse_args()
cuda_idx = str(args.CUDA)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_idx

data_loader = train_lightings_loader(args)
val_loader = val_lightings_loader(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("current device:", device)

if not os.path.exists(args.ckpt_path):
    os.makedirs(args.ckpt_path)


class Rec_RandRec_Bottleneck(): 
    def __init__(self, args):
        self.train_log_file = open(osp.join(args.ckpt_path, "training_log.txt"), "a", 1)
        self.val_log_file = open(osp.join(args.ckpt_path, "val_log.txt"), "a", 1)
        self.best_val_loss = float('inf')
        self.epochs = args.epochs
        self.epoch = 0
        self.image_size = args.image_size
        self.total_loss = 0.0
        self.val_every = 1  # every 5 epoch to check validation
        self.batch_size = args.batch_size
        # self.model = Masked_ConvAE(device)
        
        # pipe = pipe.to("cuda")
        
        self.model = Autoencoder(device)
        
        # self.model.to(device)
        # self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        self.criterion = torch.nn.MSELoss().to(device)
        
        # self.feature_loss = torch.nn.MSELoss()
        self.rand_weight = args.rand_weight
        
    def training(self):
        for self.epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_loss_rec = 0.0
            epoch_loss_rand_rec = 0.0
            epoch_loss_fc = 0.0
            epoch_loss_fu = 0.0
            
            for lightings, _ in tqdm(data_loader, desc=f'Training Epoch: {self.epoch}'):
                self.optimizer.zero_grad()
                lightings = lightings.half().to(device)
                lightings = lightings.reshape(-1, 3, args.image_size, args.image_size)
                # rec, rand_rec = self.model.rec_randrec_forward(lightings)
                rand_rec = self.model.randrec_forward()
                # loss_rec = self.criterion(lightings, rec)
                loss_rand_rec = self.criterion(lightings, rand_rec)
                # loss_ssim = self.ssim_loss(lightings, out)
                # print("fc", loss_fc)
                # print("fu", loss_fu)
                # print("rec", loss_rec)
                # print("ssim", loss_ssim)
                
                # loss = (1-self.rand_weight) * loss_rec + self.rand_weight * loss_rand_rec #+  loss_fc * 0.1 + loss_fu * 0.1
                loss = loss_rand_rec
                loss.backward()
              
                self.optimizer.step()
                # epoch_loss_rec += loss_rec.item()
                epoch_loss_rand_rec += loss_rand_rec.item()
                # epoch_loss_fc += loss_fc.item()
                # epoch_loss_fu += loss_fu.item()
                epoch_loss += loss.item()
            epoch_loss_rec /= len(data_loader)    
            epoch_loss_fc /= len(data_loader)  
            epoch_loss_fu /= len(data_loader)  
            epoch_loss /= len(data_loader)
            epoch_loss_rand_rec /= len(data_loader)
            print('Epoch {}: Loss: {:.6f}, Loss Rec: {:.6f}, Loss Rand Rec: {:.6f}, Loss fc: {:.6f}, Loss fu: {:.6f}'.format(self.epoch, epoch_loss, epoch_loss_rec, epoch_loss_rand_rec, epoch_loss_fc, epoch_loss_fu))

            if self.epoch % self.val_every == 0 or self.epoch == self.epochs - 1:
                self.model.eval()
                epoch_val_loss = 0.0
                epoch_val_loss_rec = 0.0
                epoch_val_loss_rand_rec = 0.0
                epoch_val_loss_fc = 0.0
                epoch_val_loss_fu = 0.0
                with torch.no_grad():
                    for lightings, _ in val_loader:
                        lightings = lightings.to(torch.float16).to(device)
                        lightings = lightings.reshape(-1, 3, args.image_size, args.image_size)
                        rand_rec = self.model.randrec_forward(lightings)
                        # loss_rec = self.criterion(lightings, rec)
                        loss_rand_rec = self.criterion(lightings, rand_rec)
                        # loss_ssim = self.ssim_loss(lightings, out)
               
                        # loss = (1-self.rand_weight) * loss_rec + self.rand_weight * loss_rand_rec #+  loss_fc * 0.01 + loss_fu * 0.01
                        loss = loss_rand_rec
                        # epoch_val_loss_rec += loss_rec.item()
                        epoch_val_loss_rand_rec += loss_rand_rec.item()
                        # epoch_val_loss_fc += loss_fc.item()
                        # epoch_val_loss_fu += loss_fu.item()
                        epoch_val_loss += loss.item()

                epoch_val_loss_rec /= len(val_loader)
                epoch_val_loss_rand_rec /= len(val_loader)    
                epoch_val_loss_fc /= len(val_loader)  
                epoch_val_loss_fu /= len(val_loader)  
                epoch_val_loss /= len(val_loader)

                print('Validation - Epoch {}: Loss: {:.6f}, Loss Rec: {:.6f}, Loss Rand Rec: {:.6f}, Loss fc: {:.6f}, Loss fu: {:.6f}'.format(self.epoch, epoch_val_loss, epoch_val_loss_rec, epoch_val_loss_rand_rec, epoch_val_loss_fc, epoch_val_loss_fu))
                self.val_log_file.write('Validation - Epoch {}: Loss: {:.6f}, Loss Rec: {:.6f}, Loss Rand Rec: {:.6f}, Loss fc: {:.6f}, Loss fu: {:.6f}\n'.format(self.epoch, epoch_val_loss, epoch_val_loss_rec, epoch_val_loss_rand_rec, epoch_val_loss_fc, epoch_val_loss_fu))

                if epoch_val_loss < self.best_val_loss:
                    self.best_val_loss = epoch_val_loss
                    self.save_ckpt(self.best_val_loss, "best_ckpt.pth")
                    print("Save the best checkpoint")

            self.train_log_file.write('Epoch {}: Loss: {:.6f}, Loss Rec: {:.6f}, Loss Rand Rec: {:.6f}, Loss fc: {:.6f}, Loss fu: {:.6f}\n'.format(self.epoch, epoch_loss, epoch_loss_rec, epoch_loss_rand_rec, epoch_loss_fc, epoch_loss_fu))

        self.save_ckpt(epoch_loss, "last_ckpt.pth")
        self.train_log_file.close()
        self.val_log_file.close()
        
if __name__ == '__main__':
    runner = Rec_RandRec_Bottleneck(args)
    runner.training()