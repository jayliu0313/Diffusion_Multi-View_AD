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
parser.add_argument("--learning_rate", default=2e-7)
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

# contrastive = Contrastive(args)

class Train_Base():
    def __init__(self, args):
        self.train_log_file = open(osp.join(args.ckpt_path, "training_log.txt"), "a", 1)
        self.val_log_file = open(osp.join(args.ckpt_path, "val_log.txt"), "a", 1)
        self.best_val_loss = float('inf')
        self.epochs = args.epochs
        self.epoch = 0
        self.image_size = args.image_size
        self.total_loss = 0.0
        self.val_every = 3  # every 5 epoch to check validation
        self.batch_size = args.batch_size
        # self.model = Masked_ConvAE(device)
        
        # pipe = pipe.to("cuda")
        
        self.model = Autoencoder(device)
        
        # self.model.to(device)
        # self.model.train()
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        self.criterion = torch.nn.MSELoss().to(device)
        # self.ssim_loss = kornia.losses.SSIMLoss(11, reduction='mean')
        if args.load_ckpt is not None:
            self.load_ckpt()
    
    def save_ckpt(self, curr_loss, filename):
        state_dict = {
            'model': self.model.state_dict(),
            'current_iteration': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch_loss': curr_loss,
        }
        torch.save(state_dict, os.path.join(args.ckpt_path, filename))

    def load_ckpt(self):
        checkpoint = torch.load(args.load_ckpt)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['current_iteration']
        print("Load ckpt from: ", args.load_ckpt)

    def training(self):
        raise NotImplementedError("parent class nie methods not implemented")

class Rec(Train_Base):
    def __init__(self, args):
        super().__init__(args)
        self.loss_feature = torch.nn.MSELoss()
     
    def training(self):
        for self.epoch in range(self.epochs):
            epoch_loss = 0.0
            for lightings, _ in tqdm(data_loader, desc=f'Training Epoch: {self.epoch}'):
                self.optimizer.zero_grad()
                lightings = lightings.to(device)
                lightings = lightings.reshape(-1, 3, args.image_size, args.image_size)

                out = self.model(lightings)
                rec_loss = self.criterion(lightings, out)            
                loss = rec_loss
                loss.backward()
              
                self.optimizer.step()
                epoch_loss += loss.item()
        

            epoch_loss /= len(data_loader)
            
            print('Epoch {}: Loss: {:.6f}'.format(self.epoch, epoch_loss))

            if self.epoch % self.val_every == 0 or self.epoch == self.epochs - 1:
                self.model.eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for lightings, _ in val_loader:
                        lightings = lightings.to(device)
                        lightings = lightings.reshape(-1, 3, args.image_size, args.image_size)

                        out = self.model(lightings)
                        rec_loss = self.criterion(lightings, out)            
                        loss = rec_loss

                        epoch_val_loss += loss.item()
                        
                epoch_val_loss /= len(val_loader)

                print(f"Epoch [{self.epoch}/{self.epochs}] - " f"Validation Loss: {epoch_val_loss:.6f}")
                self.val_log_file.write('Epoch {}: Loss: {:.6f}\n'.format(self.epoch, epoch_val_loss))

                if epoch_val_loss < self.best_val_loss:
                    self.best_val_loss = epoch_val_loss
                    self.save_ckpt(self.best_val_loss, "best_ckpt.pth")
                    print("Save the best checkpoint")

            self.train_log_file.write('Epoch {}: Loss: {:.6f}\n'.format(self.epoch, epoch_loss))

        self.save_ckpt(epoch_loss, "last_ckpt.pth")
        self.train_log_file.close()
        self.val_log_file.close()

class Fuse_Both_Rec(Train_Base):
    def __init__(self, args):
        super().__init__(args)

    def training(self):
        for self.epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_rec_fc_loss = 0.0
            epoch_feat_fc_loss = 0.0
            epoch_rec_fu_loss = 0.0
            for lightings, _ in tqdm(data_loader, desc=f'Training Epoch: {self.epoch}'):
                self.optimizer.zero_grad()
                
                lightings = lightings.to(device)
                in_ = lightings
                lightings = lightings.reshape(-1, 3, args.image_size, args.image_size) 
                fc, fu = self.model.encode(lightings)
                fc = fc.reshape(-1, 6, 256, 28, 28)
                fu = fu.reshape(-1, 6, 256, 28, 28)
            
                loss = 0.0
                rec_fc_loss = 0.0
                feat_fc_loss = 0.0
                rec_fu_loss = 0.0
                for i in range(6):
                    for j in range(6):
                        out = self.model.decode(fc[:, i, :, :, :], fu[:, j, :, :, :])
                        rec_fc_loss += self.criterion(in_[:, j, :, :, :], out)
                        if i != j:
                            feat_fc_loss += self.criterion(fc[:, i, :, :, :], fc[:, j, :, :, :])
                rec_fc_loss /= 36
                feat_fc_loss /= 30
                for i in range(self.batch_size):
                    for j in range(self.batch_size):
                        out = self.model.decode(fc[i, : , :, :, :], fu[j, :, :, :, :])
                        rec_fu_loss += self.criterion(in_[i, :, :, :, :], out)
                rec_fu_loss /= (self.batch_size * self.batch_size)
                loss = rec_fc_loss + feat_fc_loss + rec_fu_loss
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_rec_fc_loss += rec_fu_loss.item()
                epoch_feat_fc_loss += feat_fc_loss.item() 
                epoch_rec_fu_loss += rec_fu_loss.item()
            epoch_loss /= len(data_loader)
            epoch_rec_fc_loss /= len(data_loader)
            epoch_feat_fc_loss /= len(data_loader)
            epoch_rec_fu_loss /= len(data_loader)
            
            print('Epoch {}: Loss: {:.6f}, Rec_FC_Loss: {:.6f}, Feat_FC_Loss: {:.6f}, Rec_FU_Loss: {:.6f}'.format(self.epoch, epoch_loss, epoch_rec_fc_loss, epoch_feat_fc_loss, epoch_rec_fu_loss))
            self.train_log_file.write('Epoch {}: Loss: {:.6f}, Rec_FC_Loss: {:.6f}, Feat_FC_Loss: {:.6f}, Rec_FU_Loss: {:.6f}\n'.format(self.epoch, epoch_loss, epoch_rec_fc_loss, epoch_feat_fc_loss, epoch_rec_fu_loss))

            if self.epoch % self.val_every == 0 or self.epoch == self.epochs - 1:
                self.model.eval()
                epoch_val_loss = 0.0
                epoch_val_rec_fc_loss = 0.0
                epoch_val_feat_fc_loss = 0.0
                epoch_val_rec_fu_loss = 0.0
                with torch.no_grad():
                    for lightings, _ in val_loader:
                        lightings = lightings.to(device)
                        in_ = lightings
                        lightings = lightings.reshape(-1, 3, args.image_size, args.image_size) 
                        fc, fu = self.model.encode(lightings)
                        fc = fc.reshape(-1, 6, 256, 28, 28)
                        fu = fu.reshape(-1, 6, 256, 28, 28)
                    
                        loss = 0.0
                        rec_fc_loss = 0.0
                        feat_fc_loss = 0.0
                        rec_fu_loss = 0.0
                        for i in range(6):
                            for j in range(6):
                                out = self.model.decode(fc[:, i, :, :, :], fu[:, j, :, :, :])
                                rec_fc_loss += self.criterion(in_[:, j, :, :, :], out)
                                if i != j:
                                    feat_fc_loss += self.criterion(fc[:, i, :, :, :], fc[:, j, :, :, :])
                        rec_fc_loss /= 36
                        feat_fc_loss /= 30
                        for i in range(self.batch_size):
                            for j in range(self.batch_size):
                                out = self.model.decode(fc[i, : , :, :, :], fu[j, :, :, :, :])
                                rec_fu_loss += self.criterion(in_[i, :, :, :, :], out)
                        rec_fu_loss /= (self.batch_size * self.batch_size)
                        loss = rec_fc_loss + feat_fc_loss + rec_fu_loss

                        epoch_val_loss += loss.item()
                        epoch_val_rec_fc_loss += rec_fu_loss.item()
                        epoch_val_feat_fc_loss += feat_fc_loss.item() 
                        epoch_val_rec_fu_loss += rec_fu_loss.item()

                epoch_val_loss /= len(val_loader)
                epoch_val_rec_fc_loss /= len(val_loader)
                epoch_val_feat_fc_loss /= len(val_loader)
                epoch_val_rec_fu_loss /= len(val_loader)
                print('Validation Epoch {}: Loss: {:.6f}, Rec_FC_Loss: {:.6f}, Feat_FC_Loss: {:.6f}, Rec_FU_Loss: {:.6f}'.format(self.epoch, epoch_val_loss, epoch_val_rec_fc_loss, epoch_val_feat_fc_loss, epoch_val_rec_fu_loss))
                self.val_log_file.write('Validation Epoch {}: Loss: {:.6f}, Rec_FC_Loss: {:.6f}, Feat_FC_Loss: {:.6f}, Rec_FU_Loss: {:.6f}\n'.format(self.epoch, epoch_val_loss, epoch_val_rec_fc_loss, epoch_val_feat_fc_loss, epoch_val_rec_fu_loss))

                if epoch_val_loss < self.best_val_loss:
                    self.best_val_loss = epoch_val_loss
                    self.save_ckpt(self.best_val_loss, "best_ckpt.pth")
                    print("Save the best checkpoint")

        self.save_ckpt(epoch_loss, "last_ckpt.pth")
        self.train_log_file.close()
        self.val_log_file.close()

class Random_Rec(Train_Base):
    def __init__(self, args):
        super().__init__(args)
        # self.feature_loss = torch.nn.MSELoss()
        
    def training(self):
        for self.epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_loss_rec = 0.0
            epoch_loss_ssim = 0.0
            epoch_loss_fc = 0.0
            epoch_loss_fu = 0.0
            
            for lightings, _ in tqdm(data_loader, desc=f'Training Epoch: {self.epoch}'):
                self.optimizer.zero_grad()
                
                lightings = lightings.to(device)
                lightings = lightings.reshape(-1, 3, args.image_size, args.image_size) 
                out, loss_fc, loss_fu = self.model.rand_rec(lightings)
                loss_rec = self.criterion(lightings, out)
                loss_ssim = self.ssim_loss(lightings, out)
                # print("fc", loss_fc)
                # print("fu", loss_fu)
                # print("rec", loss_rec)
                # print("ssim", loss_ssim)
                loss =  loss_rec + loss_fc * 0.1 + loss_fu * 0.1

                loss.backward()
              
                self.optimizer.step()
                epoch_loss_rec += loss_rec.item()
                epoch_loss_ssim += loss_ssim.item()
                epoch_loss_fc += loss_fc.item()
                epoch_loss_fu += loss_fu.item()
                epoch_loss += loss.item()
            epoch_loss_rec /= len(data_loader)    
            epoch_loss_fc /= len(data_loader)  
            epoch_loss_fu /= len(data_loader)  
            epoch_loss /= len(data_loader)
            epoch_loss_ssim /= len(data_loader)
            print('Epoch {}: Loss: {:.6f}, Loss Rec: {:.6f}, Loss SSIM: {:.6f}, Loss fc: {:.6f}, Loss fu: {:.6f}'.format(self.epoch, epoch_loss, epoch_loss_rec, epoch_loss_ssim, epoch_loss_fc, epoch_loss_fu))

            if self.epoch % self.val_every == 0 or self.epoch == self.epochs - 1:
                self.model.eval()
                epoch_val_loss = 0.0
                epoch_val_loss_rec = 0.0
                epoch_val_loss_ssim = 0.0
                epoch_val_loss_fc = 0.0
                epoch_val_loss_fu = 0.0
                with torch.no_grad():
                    for lightings, _ in val_loader:
                        lightings = lightings.to(device)
                        lightings = lightings.reshape(-1, 3, args.image_size, args.image_size) 
                        out, loss_fc, loss_fu = self.model(lightings)
                        
                        loss_rec = self.criterion(lightings, out)
                        loss_ssim = self.ssim_loss(lightings, out)
                        loss = loss_rec #+ loss_fc + loss_fu
                        epoch_val_loss_rec += loss_rec.item()
                        epoch_val_loss_ssim += loss_ssim.item()
                        epoch_val_loss_fc += loss_fc.item()
                        epoch_val_loss_fu += loss_fu.item()
                        epoch_val_loss += loss.item()

                epoch_val_loss_rec /= len(val_loader)
                epoch_val_loss_ssim /= len(val_loader)    
                epoch_val_loss_fc /= len(val_loader)  
                epoch_val_loss_fu /= len(val_loader)  
                epoch_val_loss /= len(val_loader)

                print('Validation - Epoch {}: Loss: {:.6f}, Loss Rec: {:.6f}, Loss SSIM: {:.6f}, Loss fc: {:.6f}, Loss fu: {:.6f}'.format(self.epoch, epoch_val_loss, epoch_val_loss_rec, epoch_val_loss_ssim, epoch_val_loss_fc, epoch_val_loss_fu))
                self.val_log_file.write('Validation - Epoch {}: Loss: {:.6f}, Loss Rec: {:.6f}, Loss SSIM: {:.6f}, Loss fc: {:.6f}, Loss fu: {:.6f}\n'.format(self.epoch, epoch_val_loss, epoch_val_loss_rec, epoch_val_loss_ssim, epoch_val_loss_fc, epoch_val_loss_fu))

                if epoch_val_loss < self.best_val_loss:
                    self.best_val_loss = epoch_val_loss
                    self.save_ckpt(self.best_val_loss, "best_ckpt.pth")
                    print("Save the best checkpoint")

            self.train_log_file.write('Epoch {}: Loss: {:.6f}, Loss Rec: {:.6f}, Loss SSIM: {:.6f}, Loss fc: {:.6f}, Loss fu: {:.6f}\n'.format(self.epoch, epoch_loss, epoch_loss_rec, epoch_loss_ssim, epoch_loss_fc, epoch_loss_fu))

        self.save_ckpt(epoch_loss, "last_ckpt.pth")
        self.train_log_file.close()
        self.val_log_file.close()

class Rec_RandRec_Layerwise(Train_Base):
    def __init__(self, args):
        super().__init__(args)
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
                
                lightings = lightings.to(device)
                lightings = lightings.reshape(-1, 3, args.image_size, args.image_size)
               
                rec = self.model(lightings)
                rand_rec = self.model.rand_forward(lightings)
                loss_rec = self.criterion(lightings, rec)
                loss_rand_rec = self.criterion(lightings, rand_rec)
                
                loss = (1-self.rand_weight) * loss_rec + self.rand_weight * loss_rand_rec #+  loss_fc * 0.1 + loss_fu * 0.1
       
                loss.backward()
              
                self.optimizer.step()
                epoch_loss_rec += loss_rec.item()
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
                        lightings = lightings.to(device)
                        lightings = lightings.reshape(-1, 3, args.image_size, args.image_size)
                        # rec, rand_rec = self.model.rec_randrec_forward(lightings)
                        rand_rec = self.model.rand_forward(lightings)
                        rec = self.model(lightings)
                        loss_rec = self.criterion(lightings, rec)
                        loss_rand_rec = self.criterion(lightings, rand_rec)
                        # loss_ssim = self.ssim_loss(lightings, out)
               
                        loss = (1-self.rand_weight) * loss_rec + self.rand_weight * loss_rand_rec #+  loss_fc * 0.01 + loss_fu * 0.01
                        epoch_val_loss_rec += loss_rec.item()
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

class Rec_RandRec_Bottleneck(Train_Base): 
    def __init__(self, args):
        super().__init__(args)
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
                rand_rec = self.model.randrec_forward(lightings)
                # loss_rec = self.criterion(lightings, rec)
                loss_rand_rec = self.criterion(lightings, rand_rec)
                # loss_ssim = self.ssim_loss(lightings, out)
                # print("fc", loss_fc)
                # print("fu", loss_fu)
                # print("rec", loss_rec)
                # print("ssim", loss_ssim)
                
                # loss = (1-self.rand_weight) * loss_rec + self.rand_weight * loss_rand_rec #+  loss_fc * 0.1 + loss_fu * 0.1
                loss = loss_rand_rec
                print(loss)
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
                        lightings = lightings.half().to(device)
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
    # if args.training_mode == "mean":
    #     runner = Mean_Rec(args)
    # elif args.training_mode == "fuse_fc":
    #     runner = Fuse_fc_Rec(args)
    # elif args.training_mode == "fuse_both":
    #     runner = Fuse_Both_Rec(args)
    # elif args.training_mode == "random_both":
    #     runner = Random_Both_Rec(args)
    runner.training()
    