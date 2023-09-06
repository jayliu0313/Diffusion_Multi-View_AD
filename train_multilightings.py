import torch
import argparse
import os
import os.path as osp
import torch.nn.functional as F
from tqdm import tqdm
from core.data import train_lightings_loader, val_lightings_loader
from core.models.contrastive import Contrastive
from core.models.rgb_network import Masked_ConvAE, Masked_ConvAE_v2

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--data_path', default="/mnt/home_6T/public/jayliu0313/datasets/Eyecandies/", type=str)
parser.add_argument('--ckpt_path', default="./checkpoints/fuseFC_maskedConvV2Withbias_test_NoiseInput")
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--image_size', default=224, type=int)

# Training Setup
parser.add_argument("--training_mode", default="fuse_fc", help="traing type: mean, fuse_fc, fuse_both, random_both")
parser.add_argument("--model", default="Masked_Conv", help="traing type: Conv, Conv_Ins, Masked_Conv")
parser.add_argument("--load_ckpt", default=None)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
parser.add_argument("--workers", default=4)
parser.add_argument("--epochs", default=700)
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

class Train_Conv_Base():
    def __init__(self, args):
        self.train_log_file = open(osp.join(args.ckpt_path, "training_log.txt"), "a", 1)
        self.val_log_file = open(osp.join(args.ckpt_path, "val_log.txt"), "a", 1)
        self.best_val_loss = float('inf')
        self.epochs = args.epochs
        self.epoch = 0
        self.image_size = args.image_size
        self.total_loss = 0.0
        self.val_every = 5  # every 5 epoch to check validation
        self.batch_size = args.batch_size

        self.model = Masked_ConvAE_v2(device)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.criterion = torch.nn.MSELoss().to(device)
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
    
class Mean_Rec(Train_Conv_Base):
    def __init__(self, args):
        super().__init__(args)

    def training(self):
        for self.epoch in range(self.epochs):
            epoch_loss = 0.0
            for lightings,_ in tqdm(data_loader, desc=f'Training Epoch: {self.epoch}'):
                self.optimizer.zero_grad()

                lightings = lightings.reshape(-1, 3, args.image_size, args.image_size) 
                lightings = lightings.to(device)
                fc, fu = self.model.encode(lightings)
                fc = fc.reshape(-1, 6, 256, 28, 28)
                mean_fc = torch.mean(fc, dim = 1)
                fc = mean_fc.unsqueeze(1).repeat(1, 6, 1, 1, 1)
                fc = fc.reshape(-1, 256, 28, 28)
                out = self.model.decode(fc, fu)

                loss = self.criterion(lightings, out)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                
            epoch_loss /= len(data_loader)

            self.total_loss += epoch_loss
            
            print('Epoch {}: Loss: {:.6f}'.format(self.epoch, epoch_loss))

            if self.epoch % self.val_every == 0 or self.epoch == self.epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    epoch_val_loss = 0.0
                    for lightings in val_loader:
                        lightings = lightings.reshape(-1, 3, args.image_size, args.image_size) 
                        lightings = lightings.to(device)
                        fc, fu = self.model.encode(lightings)
                        fc = fc.reshape(-1, 6, 256, 28, 28)
                        mean_fc = torch.mean(fc, dim = 1)
                        fc = mean_fc.unsqueeze(1).repeat(1, 6, 1, 1, 1)
                        fc = fc.reshape(-1, 256, 28, 28)
                        out = self.model.decode(fc, fu)
                        loss = self.criterion(lightings, out)
                        epoch_val_loss += loss.item()
                    
                epoch_val_loss = epoch_val_loss / len(val_loader)
                if epoch_val_loss < self.best_val_loss:
                    self.best_val_loss = epoch_val_loss
                    self.save_ckpt(self.best_val_loss, "best_ckpt.pth")
                    print("Save the best checkpoint")

                print(f"Epoch [{self.epoch}/{self.epochs}] - " f"Valid Loss: {epoch_val_loss:.6f}")
                self.val_log_file.write('Epoch {}: Loss: {:.6f}\n'.format(self.epoch, epoch_val_loss))

            self.train_log_file.write('Epoch {}: Loss: {:.6f}\n'.format(self.epoch, epoch_loss))

        self.save_ckpt(epoch_loss, "last_ckpt.pth")
        self.train_log_file.close()
        self.val_log_file.close()

class Fuse_fc_Rec(Train_Conv_Base):
    def __init__(self, args):
        super().__init__(args)
        self.loss_feature = torch.nn.MSELoss()
        
    def training(self):
        for self.epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_rec_loss = 0.0
            epoch_fc_loss = 0.0
            for lightings, noise_imgs, _ in tqdm(data_loader, desc=f'Training Epoch: {self.epoch}'):
                self.optimizer.zero_grad()
                noise_imgs = noise_imgs.to(device)
                in_ = lightings.to(device)
                noise_imgs = noise_imgs.reshape(-1, 3, args.image_size, args.image_size) 
                fc, fu = self.model.encode(noise_imgs)
                fc = fc.reshape(-1, 6, 256, 28, 28)
                fu = fu.reshape(-1, 6, 256, 28, 28)
            
                rec_loss = 0.0
                # fc_loss = 0.0
                for i in range(6):
                    for j in range(6):
                        out = self.model.decode(fc[:, i, :, :, :], fu[:, j, :, :, :])
                        rec_loss += self.criterion(in_[:, j, :, :, :], out)
                        # if i != j:
                        #     fc_loss += self.loss_feature(fc[:, i, :, :, :], fc[:, j, :, :, :])
                # fc_loss /= 30
                # fc_loss *= 10
                rec_loss /= 36
                # print(rec_loss)
                loss = rec_loss
                loss.backward()
              
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_rec_loss += rec_loss.item() 
                # epoch_fc_loss += fc_loss.item()

            epoch_loss /= len(data_loader)
            epoch_rec_loss /= len(data_loader)
            epoch_fc_loss /= len(data_loader)
            self.total_loss += epoch_loss
            
            print('Epoch {}: Loss: {:.6f}, Rec Loss: {:.6f}, FC Loss: {:.6f}'.format(self.epoch, epoch_loss, epoch_rec_loss, epoch_fc_loss))

            if self.epoch % self.val_every == 0 or self.epoch == self.epochs - 1:
                self.model.eval()
                epoch_val_loss = 0.0
                epoch_val_rec_loss = 0.0
                epoch_val_fc_loss = 0.0
                with torch.no_grad():
                    for lightings, noise_imgs, _ in val_loader:
                        noise_imgs = noise_imgs.to(device)
                        in_ = lightings.to(device)
                        noise_imgs = noise_imgs.reshape(-1, 3, args.image_size, args.image_size) 
                        fc, fu = self.model.encode(noise_imgs)
                        fc = fc.reshape(-1, 6, 256, 28, 28)
                        fu = fu.reshape(-1, 6, 256, 28, 28)
                    
                        rec_loss = 0.0
                        # fc_loss = 0.0
                        for i in range(6):
                            for j in range(6):
                                out = self.model.decode(fc[:, i, :, :, :], fu[:, j, :, :, :])
                                rec_loss += self.criterion(in_[:, j, :, :, :], out)
                        epoch_val_rec_loss += (rec_loss.item() / 36)
                        # epoch_val_fc_loss += (fc_loss.item() / 30)
                epoch_val_rec_loss /= len(val_loader)
                epoch_val_fc_loss /= len(val_loader)
                epoch_val_loss = epoch_val_rec_loss + epoch_val_fc_loss
                print(f"Epoch [{self.epoch}/{self.epochs}] - " f"Validation Loss: {epoch_val_loss:.6f}, Rec Loss: {epoch_val_rec_loss:.6f}, FC Loss: {epoch_val_fc_loss:.6f}")
                self.val_log_file.write('Epoch {}: Loss: {:.6f}, Rec Loss: {:.6f}, FC Loss: {:.6f}\n'.format(self.epoch, epoch_val_loss, epoch_val_rec_loss, epoch_val_fc_loss))

                if epoch_val_loss < self.best_val_loss:
                    self.best_val_loss = epoch_val_loss
                    self.save_ckpt(self.best_val_loss, "best_ckpt.pth")
                    print("Save the best checkpoint")

            self.train_log_file.write('Epoch {}: Loss: {:.6f}, Rec Loss: {:.6f}, FC Loss: {:.6f}\n'.format(self.epoch, epoch_loss, epoch_rec_loss, epoch_fc_loss))

        self.save_ckpt(epoch_loss, "last_ckpt.pth")
        self.train_log_file.close()
        self.val_log_file.close()

class Fuse_Both_Rec(Train_Conv_Base):
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

class Random_Both_Rec(Train_Conv_Base):
    def __init__(self, args):
        super().__init__(args)
        self.feature_loss = torch.nn.MSELoss()
        
    def training(self):
        for self.epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_loss_rec = 0.0
            epoch_loss_fc = 0.0
            epoch_loss_fu = 0.0
            for lightings, _ in tqdm(data_loader, desc=f'Training Epoch: {self.epoch}'):
                self.optimizer.zero_grad()
                
                lightings = lightings.to(device)
                lightings = lightings.reshape(-1, 3, args.image_size, args.image_size) 
                fc, fu = self.model.encode(lightings)
                
                _, D, W, H = fc.shape
                # random fc by lightings
                fc = fc.reshape(-1, 6, D, W, H)
                mean_fc = torch.mean(fc, dim = 1)
                mean_fc = mean_fc.unsqueeze(1).repeat(1, 6, 1, 1, 1)
                loss_fc = self.feature_loss(fc, mean_fc)
                # loss_fc = 0.0
                # for i in range(6):
                #     for j in range(6):
                #         if i != j:
                #             loss_fc += self.criterion(fc[:, i, :, :, :], fc[:, j, :, :, :])        
                # loss_fc /= (36-6)
               
                random_indices = torch.randperm(6)
                fc = fc[:, random_indices, :, :]
                fc = fc.reshape(-1, D, W, H)

                # random fu by batch 
                fu = fu.reshape(-1, 6, D, W, H)
                # mean_fu = torch.mean(fu, dim = 0)
                # mean_fu = mean_fu.unsqueeze(0).repeat(fu.shape[0], 1, 1, 1, 1)
                # loss_fu = self.feature_loss(fu, mean_fu)
                # loss_fu = 0.0
                # for i in range(fu.shape[0]):
                #     for j in range(fu.shape[0]):
                #         if i != j:
                #             loss_fu += self.criterion(fu[i, :, :, :, :], fu[j, :, :, :, :])
                # loss_fu /= (fu.shape[0] * fu.shape[0] - fu.shape[0])
                random_indices = torch.randperm(fu.shape[0])
                fu = fu[random_indices]
                fu = fu.reshape(-1, D, W, H)
                
                out = self.model.decode(fc, fu)
                loss_rec = self.criterion(lightings, out)
                # print("fc", loss_fc)
                # print("fu", loss_fu)
                # print("rec", loss_rec)
                loss =  loss_rec + loss_fc

                loss.backward()
              
                self.optimizer.step()
                epoch_loss_rec += loss_rec.item()
                epoch_loss_fc += loss_fc.item()
                # epoch_loss_fu += loss_fu.item()
                epoch_loss += loss.item()
            epoch_loss_rec /= len(data_loader)    
            epoch_loss_fc /= len(data_loader)  
            epoch_loss_fu /= len(data_loader)  
            epoch_loss /= len(data_loader)
            
            print('Epoch {}: Loss: {:.6f}, Loss Rec: {:.6f}, Loss fc: {:.6f}, Loss fu: {:.6f}'.format(self.epoch, epoch_loss, epoch_loss_rec, epoch_loss_fc, epoch_loss_fu))

            if self.epoch % self.val_every == 0 or self.epoch == self.epochs - 1:
                self.model.eval()
                epoch_val_loss = 0.0
                epoch_val_loss_rec = 0.0
                epoch_val_loss_fc = 0.0
                epoch_val_loss_fu = 0.0
                with torch.no_grad():
                    for lightings, _ in val_loader:
                        lightings = lightings.to(device)
                        lightings = lightings.reshape(-1, 3, args.image_size, args.image_size) 
                        fc, fu = self.model.encode(lightings)
                        
                        _, D, W, H = fc.shape
                        # random fc by lightings
                        fc = fc.reshape(-1, 6, D, W, H)
                        mean_fc = torch.mean(fc, dim = 1)
                        mean_fc = mean_fc.unsqueeze(1).repeat(1, 6, 1, 1, 1)
                        loss_fc = self.feature_loss(fc, mean_fc)
                        # loss_fc = 0.0
                        # for i in range(6):
                        #     for j in range(6):
                        #         if i != j:
                        #             loss_fc += self.criterion(fc[:, i, :, :, :], fc[:, j, :, :, :])
                        # loss_fc /= (36-6)

                        random_indices = torch.randperm(6)
                        fc = fc[:, random_indices, :, :]
                        fc = fc.reshape(-1, D, W, H)

                        # random fu by batch 
                        fu = fu.reshape(-1, 6, D, W, H)
                        # mean_fu = torch.mean(fu, dim = 0)
                        # mean_fu = mean_fu.unsqueeze(0).repeat(fu.shape[0], 1, 1, 1, 1)
                        # loss_fu = self.feature_loss(fu, mean_fu)
                        # loss_fu = 0.0
                        # for i in range(fu.shape[0]):
                        #     for j in range(fu.shape[0]):
                        #         if i != j:
                        #             loss_fu += self.criterion(fu[i, :, :, :, :], fu[j, :, :, :, :])
                        # loss_fu /= (fu.shape[0] * fu.shape[0] - fu.shape[0])
                        random_indices = torch.randperm(fu.shape[0])
                        fu = fu[random_indices]
                        fu = fu.reshape(-1, D, W, H)
                        
                        out = self.model.decode(fc, fu)
                        loss_rec = self.criterion(lightings, out)
                        loss = loss_rec + loss_fc
                        epoch_val_loss_rec += loss_rec.item()
                        epoch_val_loss_fc += loss_fc.item()
                        # epoch_val_loss_fu += loss_fu.item()
                        epoch_val_loss += loss.item()

                epoch_val_loss_rec /= len(val_loader)    
                epoch_val_loss_fc /= len(val_loader)  
                epoch_val_loss_fu /= len(val_loader)  
                epoch_val_loss /= len(val_loader)

                print('Validation - Epoch {}: Loss: {:.6f}, Loss Rec: {:.6f}, Loss fc: {:.6f}, Loss fu: {:.6f}'.format(self.epoch, epoch_val_loss, epoch_val_loss_rec, epoch_val_loss_fc, epoch_val_loss_fu))
                self.val_log_file.write('Validation - Epoch {}: Loss: {:.6f}, Loss Rec: {:.6f}, Loss fc: {:.6f}, Loss fu: {:.6f}'.format(self.epoch, epoch_val_loss, epoch_val_loss_rec, epoch_val_loss_fc, epoch_val_loss_fu))

                if epoch_val_loss < self.best_val_loss:
                    self.best_val_loss = epoch_val_loss
                    self.save_ckpt(self.best_val_loss, "best_ckpt.pth")
                    print("Save the best checkpoint")

            self.train_log_file.write('Epoch {}: Loss: {:.6f}, Loss Rec: {:.6f}, Loss fc: {:.6f}, Loss fu: {:.6f}'.format(self.epoch, epoch_loss, epoch_loss_rec, epoch_loss_fc, epoch_loss_fu))

        self.save_ckpt(epoch_loss, "last_ckpt.pth")
        self.train_log_file.close()
        self.val_log_file.close()
 

if __name__ == '__main__':
    if args.training_mode == "mean":
        runner = Mean_Rec(args)
    elif args.training_mode == "fuse_fc":
        runner = Fuse_fc_Rec(args)
    elif args.training_mode == "fuse_both":
        runner = Fuse_Both_Rec(args)
    elif args.training_mode == "random_both":
        runner = Random_Both_Rec(args)
    runner.training()




 # for v in range(6):
            #     for w in range(6):
            #         out = model.reconstruct(fc_list[v], fu_list[w])
            #         mse_loss = criterion(lightings[:, w, :, :].to(device), out)
            #         mse_loss_batch.append(mse_loss)
            #         epoch_mse_loss += mse_loss.item()
            #         if (v < w):
            #             contrastive_loss = contrastive.loss(fc_list[v], fc_list[w])
            #             contrastive_loss_batch.append(contrastive_loss)
     
     
# def reweihgt_mse(input, target, reweight_arr):
#     return (reweight_arr * (input - target) ** 2).mean()

# def train_AE(epochs):
    # model = Autoencoder(args, device).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)
    # creteria = torch.nn.MSELoss()
    # log_file = open(osp.join(args.ckpt_path, "training_log.txt"), "a", 1)
    # total_loss = 0.0
    # for epoch in range(epochs):
    #     epoch_loss = 0.0
    #     epoch_contrastive_loss = 0.0
    #     epoch_mse_loss = 0.0
        
    #     for lightings, label in tqdm(data_loader, desc=f'Training Epoch: {epoch}'):
    #         optimizer.zero_grad()
            
    #         # Reconstruct loss
    #         lightings = lightings.reshape(-1, 3, args.image_size, args.image_size) 
    #         lightings = lightings.to(device)
    #         _, output = model(lightings)
           
    #         mse_loss = creteria(lightings, output)
    #         epoch_mse_loss += mse_loss.item()
    
    #         loss = mse_loss
    #         epoch_loss += loss.item()
            
    #         loss.backward()
    #         optimizer.step()
           
    #         torch.cuda.empty_cache()
      
    #     scheduler.step()
    #     epoch_loss /= len(data_loader)
    #     epoch_contrastive_loss /= len(data_loader)
    #     epoch_mse_loss /= len(data_loader)
        
    #     epoch_loss_list.append(epoch_loss)
    #     epoch_contrastive_loss_list.append(epoch_contrastive_loss)
    #     epoch_mse_loss_list.append(epoch_mse_loss)
        
    #     total_loss += epoch_loss
        
    #     if epoch % 50 == 0 and epoch != 0 or epoch == 20:
    #         checkpoint = {
    #             'model': model.state_dict(),
    #             'current_iteration': epoch,
    #         }
    #         torch.save(checkpoint, os.path.join(args.ckpt_path, 'model_ckpt_{:0>6d}.pth'.format(epoch)))

    #     if log_file is not None:
    #         log_file.write('Epoch {}: Loss: {:.6f} \n'.format(epoch, epoch_loss))
    #     print('Epoch {}: Loss: {:.6f}'.format(epoch, epoch_loss))

    # torch.save(checkpoint, os.path.join(args.ckpt_path, 'contrastive_ckpt_{:0>6d}.pth'.format(epoch)))
    # log_file.close()
    
# def train_mae(epochs):
#     model = mae_vit_base_patch16_dec512d8b()
    
#     checkpoint = torch.load(args.pretrained_path, map_location='cpu')

#     print("Load pre-trained checkpoint from: %s" % args.pretrained_path)
#     checkpoint_model = checkpoint['model']
#     state_dict = model.state_dict()
#     for k in ['head.weight', 'head.bias']:
#         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#             print(f"Removing key {k} from pretrained checkpoint")
#             del checkpoint_model[k]

#     # interpolate position embedding
#     interpolate_pos_embed(model, checkpoint_model)

#     # load pre-trained model
#     model.load_state_dict(checkpoint_model, strict=False)

#     model.to(device)

#     param_groups = optim_factory.param_groups_layer_decay(model, args.weight_decay)

#     optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, betas=(0.9, 0.95))
#     log_file = open(osp.join(args.ckpt_path, "training_log.txt"), "a", 1)
#     total_loss = 0.0
#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         epoch_contrastive_loss = 0.0
#         epoch_mse_loss = 0.0
        
#         for lightings, label in tqdm(data_loader, desc=f'Training Epoch: {epoch}'):
#             optimizer.zero_grad()
            
#             # Reconstructive loss
#             lightings = lightings.reshape(-1, 3, args.image_size, args.image_size) 
#             lightings = lightings.to(device)
#             loss, _, _ = model(lightings)
       
#             epoch_loss += loss.item()
            
#             loss.backward()
#             optimizer.step()
           
#             torch.cuda.empty_cache()
   
#         epoch_loss /= len(data_loader)
#         epoch_contrastive_loss /= len(data_loader)
#         epoch_mse_loss /= len(data_loader)
        
#         epoch_loss_list.append(epoch_loss)
#         epoch_contrastive_loss_list.append(epoch_contrastive_loss)
#         epoch_mse_loss_list.append(epoch_mse_loss)
        
#         total_loss += epoch_loss
        
#         if epoch % 50 == 0 and epoch != 0 or epoch == 20:
#             checkpoint = {
#                 'model': model.state_dict(),
#                 'current_iteration': epoch,
#             }
#             torch.save(checkpoint, os.path.join(args.ckpt_path, 'model_ckpt_{:0>6d}.pth'.format(epoch)))

#         if log_file is not None:
#             log_file.write('Epoch {}: Loss: {:.6f} \n'.format(epoch, epoch_loss))
#         print('Epoch {}: Loss: {:.6f}'.format(epoch, epoch_loss))

#     torch.save(checkpoint, os.path.join(args.ckpt_path, 'contrastive_ckpt_{:0>6d}.pth'.format(epoch)))
#     log_file.close()
