import torch
import argparse
import os
import os.path as osp
from tqdm import tqdm
from core.data import train_lightings_loader
from core.models.contrastive import Contrastive
from core.models.network import Convolution_AE, Autoencoder
from core.models.ResNetAE import ResNetAE
from core.models.unet_model import UNet
from core.models.mae import mae_vit_base_patch16_dec512d8b
from utils.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
import timm.optim.optim_factory as optim_factory

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--data_path', default="/mnt/home_6T/public/jayliu0313/datasets/Eyecandies/", type=str)
parser.add_argument('--ckpt_path', default="./checkpoints/cnn_fuseRec_lr00003")
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--image_size', default=224, type=int)
# Contrstive Learning
parser.add_argument("--contrastive_w", default=0.001)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
# Training Setup
parser.add_argument("--pretrained_path", default="checkpoints/pretrained_vitB/mae_pretrain_vit_base.pth")
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
parser.add_argument("--workers", default=4)
parser.add_argument("--epochs", default=600)
parser.add_argument("--common_feature_dim", default=256)
parser.add_argument("--unique_feature_dim", default=256)
parser.add_argument('--CUDA', type=int, default=0, help="choose the device of CUDA")


args = parser.parse_args()
cuda_idx = str(args.CUDA)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_idx

data_loader = train_lightings_loader(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("current device:", device)

if not os.path.exists(args.ckpt_path):
    os.makedirs(args.ckpt_path)

contrastive = Contrastive(args)


epoch_loss_list = []
epoch_contrastive_loss_list = []
epoch_mse_loss_list = []

def train_mean_reconsturct(epochs):
    model = Convolution_AE(args, device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss().to(device)
    log_file = open(osp.join(args.ckpt_path, "training_log.txt"), "a", 1)
    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0

        for lightings, label in tqdm(data_loader, desc=f'Training Epoch: {epoch}'):
            optimizer.zero_grad()

            lightings = lightings.reshape(-1, 3, args.image_size, args.image_size) 
            lightings = lightings.to(device)
            fc, fu = model.get_feature(lightings)
            fc = fc.reshape(-1, 6, 256, 28, 28)
            mean_fc = torch.mean(fc, dim = 1)
            fc = mean_fc.unsqueeze(1).repeat(1, 6, 1, 1, 1)
            fc = fc.reshape(-1, 256, 28, 28)
            out = model.reconstruct(fc, fu)
            loss = criterion(lightings, out)

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            # torch.cuda.empty_cache()
        
        epoch_loss /= len(data_loader)
        
        epoch_loss_list.append(epoch_loss)
        
        total_loss += epoch_loss
        
        if epoch % 50 == 0 and epoch != 0 or epoch == 20:
            checkpoint = {
                'model': model.state_dict(),
                'current_iteration': epoch,
            }
            torch.save(checkpoint, os.path.join(args.ckpt_path, 'ckpt_{:0>6d}.pth'.format(epoch)))
        
        if log_file is not None:
            log_file.write('Epoch {}: Loss: {:.6f}\n'.format(epoch, epoch_loss))
        print('Epoch {}: Loss: {:.6f}'.format(epoch, epoch_loss))

    torch.save(checkpoint, os.path.join(args.ckpt_path, 'ckpt_{:0>6d}.pth'.format(epoch)))
    log_file.close()

def train_fuse_reconsturct(epochs):
    model = Convolution_AE(args, device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss().to(device)
    log_file = open(osp.join(args.ckpt_path, "training_log.txt"), "a", 1)
    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0

        for lightings, label in tqdm(data_loader, desc=f'Training Epoch: {epoch}'):
            optimizer.zero_grad()
            
            in_ = lightings.to(device)
            lightings = lightings.reshape(-1, 3, args.image_size, args.image_size) 
            lightings = lightings.to(device)
            fc, fu = model.get_feature(lightings)
            fc = fc.reshape(-1, 6, 256, 28, 28)
            fu = fu.reshape(-1, 6, 256, 28, 28)
        
            loss = 0
            for i in range(6):
                for j in range(6):
                    out = model.reconstruct(fc[:, i, :, :, :], fu[:, j, :, :, :])
                    loss += criterion(in_[:, j, :, :, :], out)
            loss /= 6
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            # torch.cuda.empty_cache()

        epoch_loss /= len(data_loader)
        
        epoch_loss_list.append(epoch_loss)
        
        total_loss += epoch_loss
        
        if epoch % 50 == 0 and epoch != 0 or epoch == 20:
            checkpoint = {
                'model': model.state_dict(),
                'current_iteration': epoch,
            }
            torch.save(checkpoint, os.path.join(args.ckpt_path, 'ckpt_{:0>6d}.pth'.format(epoch)))
        
        if log_file is not None:
            log_file.write('Epoch {}: Loss: {:.6f}\n'.format(epoch, epoch_loss))
        print('Epoch {}: Loss: {:.6f}'.format(epoch, epoch_loss))

    torch.save(checkpoint, os.path.join(args.ckpt_path, 'ckpt_{:0>6d}.pth'.format(epoch)))
    log_file.close()


def main():
    train_fuse_reconsturct(args.epochs)
main()



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
