import torch
import argparse
import numpy as np
import os
from torchvision import transforms
from tqdm import tqdm
from utils.au_pro_util import calculate_au_pro
from utils.visualize_util import *
from utils.utils import KNNGaussianBlur
from sklearn.metrics import roc_auc_score


from core.models.autoencoder import Autoencoder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
class Base_Method():
    def __init__(self, args, cls_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.load_model(args.rgb_ckpt_path, args.nmap_ckpt_path)
        self.initialize_score()

        # if args.load_decomp_ckpt is not None:
        #     checkpoint_dict = torch.load(args.load_decomp_ckpt, map_location=self.device)
        #     # Load VAE checkpoint  
        #     if checkpoint_dict['backbone'] is not None:
        #         print("load vae checkpoints!")
        #         self.vae.load_state_dict(checkpoint_dict['backbone'])

            # if  checkpoint_dict['decomp_block'] is not None:
            #     print("load decomp checkpoints!")
            #     self.decomp_block.load_state_dict(checkpoint_dict['decomp_block'])
                
        # Load vae model
        
        self.blur = KNNGaussianBlur()
        # AE = Autoencoder(self.device)
        # AE.load_state_dict(torch.load(args.load_vae_ckpt_path, map_location=self.device)['model'])
        # self.vae = AE.vae
        # self.decomp_block = AE.decomp_block

        # self.vae.requires_grad_(False)

        
        
        self.criteria = torch.nn.MSELoss()    
        # self.fc_dim = args.common_feature_dim
        self.patch_lib = []
        self.image_size = args.image_size
        
        self.cls_path = cls_path
        self.cls_rec_loss = 0.0
        self.reconstruct_path = os.path.join(cls_path, "Reconstruction")
        self.score_type = args.score_type
        
        self.pdist = torch.nn.PairwiseDistance(p=2, eps= 1e-12)
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.resize = torch.nn.AdaptiveAvgPool2d((28, 28))
        # self.blur = torch.Gua(4).to(self.device)

        self.image_transform = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        if not os.path.exists(self.reconstruct_path):
            os.makedirs(self.reconstruct_path)
            
    def initialize_score(self):
        self.image_list = list()
        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.predictions = []
        self.gts = []

    def add_sample_to_mem_bank(self, lightings):
        raise NotImplementedError
    
    def predict(self, item, lightings, gt, label):
        raise NotImplementedError
    
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
    
    def calculate_metrics(self):

        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.stack(self.pixel_preds)
        self.pixel_labels = np.stack(self.pixel_labels)

        flatten_pixel_preds = self.pixel_preds.flatten()
        flatten_pixel_labels = self.pixel_labels.flatten()

        gts = np.array(self.pixel_labels).reshape(-1, self.image_size, self.image_size)
        predictions =  self.pixel_preds.reshape(-1, self.image_size, self.image_size)

        self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
        self.pixel_rocauc = roc_auc_score(flatten_pixel_labels, flatten_pixel_preds)
        self.au_pro, _ = calculate_au_pro(gts, predictions)

        return self.image_rocauc, self.pixel_rocauc, self.au_pro
    
    def get_rec_loss(self):
        return self.cls_rec_loss
    
    def visualizae_heatmap(self):
        score_map = self.pixel_preds.reshape(-1, self.image_size, self.image_size)
        gt_mask = np.squeeze(np.array(self.pixel_labels, dtype=np.bool), axis=1)
        
        visualization(self.image_list, self.image_labels, self.image_preds, gt_mask, score_map, self.cls_path)
        
        
    # def load_model(self, rgb_ckpt_path=None, nmap_ckpt_path=None):
    #     if nmap_ckpt_path != None:
    #         print("Load the checkpoint of normal map model...")
    #         self.nmap_model = NMap_AE(self.device)
    #         self.nmap_model.to(self.device)
    #         self.nmap_model.eval()
    #         checkpoint = torch.load(nmap_ckpt_path, map_location=self.device)
    #         self.nmap_model.load_state_dict(checkpoint['model'])
    #         self.nmap_model.freeze_model()
    #     if rgb_ckpt_path != None:
    #         print("Load the checkpoint of rgb model...")
    #         self.rgb_model = Autoencoder(self.device)
    #         self.rgb_model.to(self.device)
    #         self.rgb_model.eval()
    #         checkpoint = torch.load(rgb_ckpt_path, map_location=self.device)
    #         self.rgb_model.load_state_dict(checkpoint['model'])
    #         self.rgb_model.freeze_model()
    