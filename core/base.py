import torch
import argparse
import numpy as np
import os
from tqdm import tqdm
from utils.au_pro_util import calculate_au_pro
from utils.visualize_util import *
from sklearn.metrics import roc_auc_score
from core.models.rgb_network import Convolution_AE_v2, Convolution_AE
from core.models.nmap_network import NMap_AE

class Base_Method():
    def __init__(self, args, cls_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model(args.method_name, args.ckpt_path)
        self.initialize_score()

        self.criteria = torch.nn.MSELoss()    
        # self.fc_dim = args.common_feature_dim
        self.patch_lib = []
        self.image_size = args.image_size
        
        self.cls_path = cls_path
        self.cls_rec_loss = 0.0
        self.reconstruct_path = os.path.join(cls_path, "Reconstruction")
        self.score_type = args.score_type

        if not os.path.exists(self.reconstruct_path):
            os.makedirs(self.reconstruct_path)

    def load_model(self, method_name, ckpt_path):
        if method_name == "nmap_rec":
            self.model = NMap_AE(self.device)
        else:
            self.model = Convolution_AE_v2(self.device)
        self.model.to(self.device)
        self.model.eval()
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.freeze_model()

    def initialize_score(self):
        self.image_list = list()
        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.predictions = []
        self.gts = []

    def add_sample_to_mem_bank(self, lightings):
        pass
    
    def predict(self, item, lightings, gt, label):
        raise NotImplementedError

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
    