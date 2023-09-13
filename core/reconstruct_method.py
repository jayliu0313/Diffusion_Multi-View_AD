import torch
import numpy as np
from core.base import Base_Method
from utils.visualize_util import display_one_img, display_image, display_mean_fusion
from utils.utils import t2np
from patchify import patchify

class Base_Reconstruct(Base_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)

    def compute_score(self, score_maps, top_k = 4):
        six_map_patches = []
        for i in range(score_maps.shape[0]):
            patches = patchify(t2np(score_maps[i, :, :, :].permute(2, 1, 0)), (8, 8, 1), 2)
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
    
# test method 1
class Mean_Rec(Base_Reconstruct):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
    
    def predict(self, item, lightings, gt, label):
        lightings = lightings.squeeze().to(self.device)
        fc, fu =  self.model.encode(lightings)
        fc_mean = torch.mean(fc, dim=0)
        fc_mean = fc_mean.unsqueeze(0).repeat(6, 1, 1, 1)
        out = self.model.decode(fc_mean, fu)
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

# test method 2
class Rec(Base_Reconstruct):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
    
    def predict(self, item, lightings, gt, label):
        lightings = lightings.squeeze().to(self.device)
        out, _ =  self.model(lightings)

        lightings = self.average(lightings)
        out = self.average(out)
        
        loss = self.criteria(lightings, out)
        
        self.cls_rec_loss += loss.item()
        score_maps = torch.sum(torch.abs(lightings - out), dim=1)
        # score_maps = self.blur(score_maps)
        score_maps = score_maps.unsqueeze(1)
        # final_score = self.compute_score(score_maps)
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
        # display_image(t2np(lightings), t2np(out), self.reconstruct_path, item)

# test method 3
class Recursive_Rec(Base_Reconstruct):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.times = 5

    def predict(self, item, lightings, gt, label):
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

# test method 4
class Nmap_Rec(Base_Reconstruct):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
    
    def predict(self, item, normal, gt, label):
        normal = normal.to(self.device)
        out = self.model(normal)
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

class RGB_Nmap_Rec(Base_Reconstruct):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
    
    def predict(self, item, lightings, Nmap, gt, label):
        lightings = lightings.squeeze().to(self.device)
        out, _ =  self.model(lightings)

        lightings = self.average(lightings)
        out = self.average(out)
        
        loss = self.criteria(lightings, out)
        
        self.cls_rec_loss += loss.item()
        score_maps = torch.sum(torch.abs(lightings - out), dim=1)
        # score_maps = self.blur(score_maps)
        score_maps = score_maps.unsqueeze(1)
        # final_score = self.compute_score(score_maps)
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
        # display_image(t2np(lightings), t2np(out), self.reconstruct_path, item)

  