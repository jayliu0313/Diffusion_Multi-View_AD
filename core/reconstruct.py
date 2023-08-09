import torch
import argparse
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.au_pro_util import calculate_au_pro
from utils.visualize_util import *
from sklearn.metrics import roc_auc_score

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


class Base_Reconsturct():
    def __init__(self, args, model, cls_path):
        self.model = model
        self.image_list = list()
        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.predictions = []
        self.gts = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc_dim = args.common_feature_dim
        
        self.image_size = args.image_size
        
        self.cls_path = cls_path
        self.reconstruct_path = os.path.join(cls_path, "Reconsturction")
        self.score_type = args.score_type

        if not os.path.exists(self.reconstruct_path):
            os.makedirs(self.reconstruct_path)

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
    
    def visualizae_heatmap(self):
        score_map = self.pixel_preds.reshape(-1, self.image_size, self.image_size)
        gt_mask = np.squeeze(np.array(self.pixel_labels, dtype=np.bool), axis=1)
        
        visualization(self.image_list, self.image_labels, self.image_preds, gt_mask, score_map, self.cls_path)
    

# test method 1
class Mean_Reconstruct(Base_Reconsturct):
    def __init__(self, args, model, cls_path):
        super().__init__(args, model, cls_path)
    
    def predict(self, item, lightings, gt, label):
        lightings = lightings.squeeze().to(self.device)
        fc, fu =  self.model.get_feature(lightings)
        fc_mean = torch.mean(fc, dim=0)
        fc_mean = fc_mean.unsqueeze(0).repeat(6, 1, 1, 1)
        out =  self.model.reconstruct(fc_mean, fu)
        score_maps = torch.sum(torch.abs(lightings - out), dim=1)
        score_maps = score_maps.unsqueeze(1)

        if(self.score_type == 0):
            final_score = -99999
            final_map = torch.zeros((1, self.image_size, self.image_size))
            img = torch.zeros((3, self.image_size, self.image_size))
            for i in range(6):
                score_map = score_maps[i, :, :, :]
                topk_score, _ = torch.topk(score_map.flatten(), 20)
                score = torch.mean(topk_score)
                if(final_score < score):
                    final_score = score  
                    final_map = score_map
                    img = lightings[i, :, :, :]
        elif(self.score_type == 1):
            img = lightings[0, :, :, :]
            final_map = torch.mean(score_maps, dim=0)
            topk_score, _ = torch.topk(final_map.flatten(), 20)
            final_score = torch.mean(topk_score)

        self.image_labels.append(np.array(label))
        self.image_preds.append(t2np(final_score))
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(final_map))
        self.pixel_labels.extend(t2np(gt))

        if item % 5 == 0:
            display_mean_fusion(t2np(lightings), t2np(out), self.reconstruct_path, item)

# test method 2
class Reconstruct(Base_Reconsturct):
    def __init__(self, args, model, cls_path):
        super().__init__(args, model, cls_path)
    
    def predict(self, item, lightings, gt, label):
        lightings = lightings.squeeze().to(self.device)
        # _, out =  self.model(lightings)
        _, out, _ =  self.model(lightings)
        out = self.model.unpatchify(out)
        print(out.shape)
        score_maps = torch.sum(torch.abs(lightings - out), dim=1)
        score_maps = score_maps.unsqueeze(1)

        if(self.score_type == 0):
            final_score = -99999
            final_map = torch.zeros((1, self.image_size, self.image_size))
            img = torch.zeros((3, self.image_size, self.image_size))
            for i in range(6):
                score_map = score_maps[i, :, :, :]
                topk_score, _ = torch.topk(score_map.flatten(), 20)
                score = torch.mean(topk_score)
                if(final_score < score):
                    final_score = score  
                    final_map = score_map
                    img = lightings[i, :, :, :]
        elif(self.score_type == 1):
            img = lightings[0, :, :, :]
            final_map = torch.mean(score_maps, dim=0)
            topk_score, _ = torch.topk(final_map.flatten(), 20)
            final_score = torch.mean(topk_score)

        self.image_labels.append(label)
        self.image_preds.append(t2np(final_score))
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(final_map))
        self.pixel_labels.extend(t2np(gt))

        if item % 5 == 0:
            display_image(t2np(lightings), t2np(out), self.reconstruct_path, item)