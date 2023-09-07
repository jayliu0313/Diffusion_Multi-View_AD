import torch
from core.base import Base_Method
from utils.visualize_util import display_one_img, display_image, display_mean_fusion
from utils.utils import t2np

class Base_Reconstruct(Base_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        
    def compute_max_score(self, score_maps, lightings):
        # print(lightings.shape)
        # lightings = gauss_noise_tensor(lightings)
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
        return final_map, final_score, img
    
    def compute_mean_score(self, score_maps, lightings):
        img = lightings[0, :, :, :]
        final_map = torch.mean(score_maps, dim=0)
        topk_score, _ = torch.topk(final_map.flatten(), 20)
        final_score = torch.mean(topk_score)
        return final_map, final_score, img
    
# test method 1
class Mean_Reconstruct(Base_Reconstruct):
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
            final_map, final_score, img = self.compute_max_score(score_maps, lightings)
        elif(self.score_type == 1):
            final_map, final_score, img = self.compute_mean_score(score_maps, lightings)
        
        self.image_labels.append(label.numpy())
        self.image_preds.append(t2np(final_score))
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(final_map))
        self.pixel_labels.extend(t2np(gt))

       
        display_mean_fusion(t2np(lightings), t2np(out), self.reconstruct_path, item)

# test method 2
class Reconstruct(Base_Reconstruct):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
    
    def predict(self, item, lightings, gt, label):
        lightings = lightings.squeeze().to(self.device)
        _, out =  self.model(lightings)
        loss = self.criteria(lightings, out)
        self.cls_rec_loss += loss.item()
        score_maps = torch.sum(torch.abs(lightings - out), dim=1)
        score_maps = score_maps.unsqueeze(1)

        if(self.score_type == 0):
            final_map, final_score, img = self.compute_max_score(score_maps, lightings)
        elif(self.score_type == 1):
            final_map, final_score, img = self.compute_mean_score(score_maps, lightings)

        self.image_labels.append(label)
        self.image_preds.append(t2np(final_score))
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(final_map))
        self.pixel_labels.extend(t2np(gt))

        if item % 5 == 0:
            display_image(t2np(lightings), t2np(out), self.reconstruct_path, item)

# test method 3
class Normal_Reconstruct(Base_Reconstruct):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
    
    def predict(self, item, normal, gt, label):
        normal = normal.to(self.device)
        out = self.model(normal)
        loss = self.criteria(normal, out)
        self.cls_rec_loss += loss.item()
        score_map = torch.sum(torch.abs(normal - out), dim=1)
        topk_score, _ = torch.topk(score_map.flatten(), 20)
        final_score = torch.mean(topk_score)
        self.image_labels.append(label)
        self.image_preds.append(t2np(final_score))
        self.image_list.append(t2np(normal.squeeze()))
        self.pixel_preds.append(t2np(score_map.squeeze()))
        self.pixel_labels.extend(t2np(gt))

        # if item % 5 == 0:
        display_one_img(t2np(normal.squeeze()), t2np(out.squeeze()), self.reconstruct_path, item)


  