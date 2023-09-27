import torch
import numpy as np
from core.base import Base_Method
from utils.visualize_util import display_one_img, display_image, display_mean_fusion
from utils.utils import t2np
from patchify import patchify

class Base_Reconstruct(Base_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)

    def compute_score(self, score_maps, top_k = 1):
        six_map_patches = []
        for i in range(score_maps.shape[0]):
            patches = patchify(t2np(score_maps[i, :, :, :].permute(2, 1, 0)), (8, 8, 1), 8)
            patches_score = torch.tensor(np.mean(patches, axis=(3, 4)).flatten())
            six_map_patches.append(patches_score)
        six_map_patches = torch.stack(six_map_patches)
        mean_map_patches = torch.mean(six_map_patches, dim=0)
        top_scores, _ = torch.topk(mean_map_patches, top_k)
        final_score = torch.mean(top_scores)
        return final_score

    def compute_max_smap(self, score_maps, lightings):
        final_map = torch.zeros((1, self.image_size, self.image_size))
        img = torch.zeros((3, self.image_size, self.image_size))
        final_map, _ = torch.max(score_maps, 0)
        topk_score, _ = torch.topk(final_map.flatten(), 25)
        final_score = torch.mean(topk_score)
        img = lightings[5, :, :, :]
        return final_map, final_score, img
    
    def compute_mean_smap(self, score_maps, lightings):
        img = lightings[0, :, :, :]
        final_map = torch.mean(score_maps, dim=0)
        topk_score, _ = torch.topk(final_map.flatten(), 25)
        final_score = torch.mean(topk_score)
        return final_map, final_score, img
    
# test method 1: mean fc to reconstruct each image
class Mean_Rec(Base_Reconstruct):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
    
    def predict(self, item, lightings, _, gt, label):
        lightings = lightings.squeeze().to(self.device)
        out =  self.rgb_model.mean_rec(lightings)
        loss = self.criteria(lightings, out)
        self.cls_rec_loss += loss.item()

        score_maps = torch.sum(torch.abs(lightings - out), dim=1)
        score_maps = score_maps.unsqueeze(1)

        if(self.score_type == 0):
            final_map, final_score, img = self.compute_max_smap(score_maps, lightings)
        elif(self.score_type == 1):
            final_map, final_score, img = self.compute_mean_smap(score_maps, lightings)
        
        self.image_labels.append(label.numpy())
        self.image_preds.append(t2np(final_score))
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(final_map))
        self.pixel_labels.extend(t2np(gt))

       
        display_mean_fusion(t2np(lightings), t2np(out), self.reconstruct_path, item)

# test method 2: reconstruct each image (using individual fc and fu)
class Rec(Base_Reconstruct):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
    
    def predict(self, item, lightings, _, gt, label):
        lightings = lightings.squeeze().to(self.device)
        out =  self.rgb_model.rec(lightings)

        lightings = self.average(lightings)
        out = self.average(out)
        
        loss = self.criteria(lightings, out)
        
        self.cls_rec_loss += loss.item()
        score_maps = torch.sum(torch.abs(lightings - out), dim=1)
        score_maps = score_maps.unsqueeze(1)
        final_score = self.compute_score(score_maps)
        if(self.score_type == 0):
            final_map, _, img = self.compute_max_smap(score_maps, lightings)
        elif(self.score_type == 1):
            final_map, _, img = self.compute_mean_smap(score_maps, lightings)
        
        self.image_labels.append(label)
        self.image_preds.append(t2np(final_score))
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(final_map))
        self.pixel_labels.extend(t2np(gt))

        # if item % 5 == 0:
        # display_image(t2np(lightings), t2np(out), self.reconstruct_path, item)

# test method 3: not use
class Recursive_Rec(Base_Reconstruct):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.times = 5

    def predict(self, item, lightings, _, gt, label):
        lightings = lightings.squeeze().to(self.device)
        in_ = lightings
        for _ in range(self.times):
            out, _ =  self.model(in_)
            in_ = out
        out = self.average(out)
        lightings = self.average(lightings)
        loss = self.criteria(lightings, out)
        self.cls_rec_loss += loss.item()
        score_maps = torch.sum(torch.abs(lightings - out), dim=1)
        score_maps = score_maps.unsqueeze(1)

        if(self.score_type == 0):
            final_map, final_score, img = self.compute_max_smap(score_maps, lightings)
        elif(self.score_type == 1):
            final_map, final_score, img = self.compute_mean_smap(score_maps, lightings)

        self.image_labels.append(label)
        self.image_preds.append(t2np(final_score))
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(final_map))
        self.pixel_labels.extend(t2np(gt))

        # if item % 5 == 0:
        display_image(t2np(lightings), t2np(out), self.reconstruct_path, item)

# normal map reconstruction
class Nmap_Rec(Base_Reconstruct):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
    
    def predict(self, item, _, normal, gt, label):
        normal = normal.to(self.device)
        out = self.nmap_model(normal)
        normal = self.average(normal)
        out = self.average(out)
        loss = self.criteria(normal, out)
        self.cls_rec_loss += loss.item()
        score_map = torch.sum(torch.abs(normal - out), dim=1)
        score_map = score_map.unsqueeze(0)
        final_score = self.compute_score(score_map)
        # topk_score, _ = torch.topk(score_map.flatten(), 20)
        # final_score = torch.mean(topk_score)
        self.image_labels.append(label)
        self.image_preds.append(t2np(final_score))
        self.image_list.append(t2np(normal.squeeze()))
        self.pixel_preds.append(t2np(score_map.squeeze()))
        self.pixel_labels.extend(t2np(gt))

        # if item % 5 == 0:
        display_one_img(t2np(normal.squeeze()), t2np(out.squeeze()), self.reconstruct_path, item)

# test method 5: normal map and image reconstruction
class RGB_Nmap_Rec(Base_Reconstruct):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
    
    def predict(self, item, lightings, nmap, gt, label):
        lightings = lightings.squeeze().to(self.device)
        
        rgb_out =  self.rgb_model.rec(lightings)
        
        lightings = self.average(lightings)
        rgb_out = self.average(rgb_out)
        loss = self.criteria(lightings, rgb_out)
        self.cls_rec_loss += loss.item()
        rgb_score_maps = torch.sum(torch.abs(lightings - rgb_out), dim=1)
        rgb_score_maps = rgb_score_maps.unsqueeze(1)
        rgb_final_score = self.compute_score(rgb_score_maps)
        if(self.score_type == 0):
            rgb_score_maps, _, img = self.compute_max_smap(rgb_score_maps, lightings)
        elif(self.score_type == 1):
            rgb_score_maps, _, img = self.compute_mean_smap(rgb_score_maps, lightings)
        
        nmap = nmap.to(self.device)
        nmap_out = self.nmap_model(nmap)
        nmap = self.average(nmap)
        nmap_out = self.average(nmap_out)
        nmap_score_maps = torch.sum(torch.abs(nmap - nmap_out), dim=1)
        # print(nmap_score_maps.shape)
        # topk_score, _ = torch.topk(nmap_score_maps.flatten(), 20)
        # final_nmap_score = torch.mean(topk_score)
        final_nmap_score = self.compute_score(nmap_score_maps.unsqueeze(0))

        final_map = torch.maximum(nmap_score_maps, rgb_score_maps)
        final_score = rgb_final_score * final_nmap_score
        self.image_labels.append(label)
        self.image_preds.append(t2np(final_score))
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(final_map))
        self.pixel_labels.extend(t2np(gt))

        if item % 5 == 0:
            display_image(t2np(lightings), t2np(rgb_out), self.reconstruct_path, item)

  