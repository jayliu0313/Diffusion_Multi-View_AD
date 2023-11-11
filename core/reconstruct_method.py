import torch
import numpy as np
from core.base import Base_Method
from core.models.network_util import Decom_Block
from utils.visualize_util import display_one_img, display_image, display_mean_fusion
from utils.utils import t2np
from patchify import patchify
from diffusers import AutoencoderKL

class Reconstruct_Method(Base_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        # Load vae model
        self.vae = AutoencoderKL.from_pretrained(
                    args.diffusion_id,
                    subfolder="vae",
                    revision=args.revision,
                    torch_dtype=torch.float32
                ).to(self.device)
        self.decomp_block = Decom_Block(4).to(self.device)
        
        # Load checkpoint  
        self.load_ckpt(args.load_vae_ckpt, args.load_decom_ckpt)
        self.vae.requires_grad_(False)
        self.decomp_block.requires_grad_(False)
        
    def load_ckpt(self, vae_ckpt, decomp_ckpt):
        if vae_ckpt is not None:
            self.vae.load_state_dict(torch.load(vae_ckpt, map_location=self.device))
        if decomp_ckpt is not None:
            self.decomp_block.load_state_dict(torch.load(decomp_ckpt, map_location=self.device))
            
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
    
    def get_pretrained_feature(self, x):
        rgb_feature_maps = self.rgb_model.get_layer_feature(x)
        rgb_resized_maps = [self.resize(self.average(fmap)) for fmap in rgb_feature_maps]
        rgb_patch = torch.cat(rgb_resized_maps, 1)
        return rgb_patch
        
    def compute_feature_dist(self, feat1, feat2):
        feat1 = feat1.permute(0, 2, 3, 1)
        feat2 = feat2.permute(0, 2, 3, 1)
        s_map_size28 = self.pdist(feat1, feat2)
        s_map_size28 = torch.mean(s_map_size28, dim=0)
        s = torch.max(s_map_size28)
        s_map_size28 = s_map_size28.view(1, 1, 28, 28)
        s_map_size_img = torch.nn.functional.interpolate(s_map_size28, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        return s_map_size_img.to('cpu'), s.to('cpu')
    
    def compute_score(self, score_maps, top_k = 1):
        six_map_patches = []
        for i in range(score_maps.shape[0]):
            patches = patchify(t2np(score_maps[i, :, :, :].permute(2, 1, 0)), (8, 8, 1), 4)
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
    
class Nmap_Rec(Reconstruct_Method):
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

class RGB_Nmap_Rec(Reconstruct_Method):
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

class Nmap_Repair(Reconstruct_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        
    def predict(self, item, lightings, nmap, gt, label):
        nmap = nmap.to(self.device)
        feat = self.nmap_model.encode(nmap)
        rep_feat = self.nmap_model.Repair_Feat(feat)
        
        score_map = torch.sum(torch.abs(feat - rep_feat), dim=1)
        score_map = score_map.unsqueeze(0)
        score_map = torch.nn.functional.interpolate(score_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners = False)
        topk_score = torch.max(score_map.flatten())

        rec = self.nmap_model.decode(rep_feat)
        loss = self.criteria(feat, rep_feat)
        self.cls_rec_loss += loss.item()
        self.image_labels.append(label)
        self.image_preds.append(t2np(topk_score))
        self.image_list.append(t2np(nmap.squeeze()))
        self.pixel_preds.append(t2np(score_map.squeeze()))
        self.pixel_labels.extend(t2np(gt))
        if item % 5 == 0:
            display_one_img(t2np(nmap.squeeze()), t2np(rec.squeeze()), self.reconstruct_path, item)

class Vae_Rec(Reconstruct_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
                
    def predict(self, item, lightings, _, gt, label):
        lightings = lightings.squeeze().to(self.device)
        
        latents = self.image2latents(lightings)
        # mean fc reconstruction
        mean_fc = self.decomp_block.get_meanfc(latents)
        fu = self.decomp_block.get_fu(latents)
        latents = self.decomp_block.fuse_both(mean_fc, fu)
        # print(latents.shape)
        # own fc reconstruction
        # latents = self.decomp_block(latents)
        out = self.latents2image(latents)

        # print(out.min())
        # print(out.max())
        if False:
            # img_avg = torch.nn.AvgPool2d(3, stride=1, padding=1)
            # imgnet_in = img_avg(lightings)
            # imgnet_out = img_avg(out)
    
            in_feat = self.get_pretrained_feature(lightings)
            repaired_feat = self.get_pretrained_feature(out)
            final_map, final_score = self.compute_feature_dist(in_feat, repaired_feat)
            out_avg = self.average(out)
            lightings_avg = self.average(lightings)
            loss = self.criteria(lightings_avg, out_avg)
        else:
            img_avg = torch.nn.AvgPool2d(3, stride=1, padding=1)
            out = img_avg(out)
            lightings = img_avg(lightings)
            loss = self.criteria(lightings, out)
            score_maps = torch.sum(torch.abs(lightings - out), dim=1)
            score_maps = score_maps.unsqueeze(1)
            final_score = self.compute_score(score_maps)
            if(self.score_type == 0):
                final_map, _, img = self.compute_max_smap(score_maps, lightings)
            elif(self.score_type == 1):
                final_map, _, img = self.compute_mean_smap(score_maps, lightings)
            # if item % 2 == 0:
            display_image(t2np(lightings), t2np(out), self.reconstruct_path, item)
            
        self.cls_rec_loss += loss.item()
        self.image_labels.append(label)
        self.image_preds.append(t2np(final_score))
        self.image_list.append(t2np(lightings[5, :, :, :]))
        self.pixel_preds.append(t2np(final_map))
        self.pixel_labels.extend(t2np(gt))


# class Mean_Rec(Reconstruct_Method):
#     def __init__(self, args, cls_path):
#         super().__init__(args, cls_path)
    
#     def predict(self, item, lightings, _, gt, label):
#         lightings = lightings.squeeze().to(self.device)
#         out =  self.rgb_model.mean_rec(lightings)
#         loss = self.criteria(lightings, out)
#         self.cls_rec_loss += loss.item()

#         score_maps = torch.sum(torch.abs(lightings - out), dim=1)
#         score_maps = score_maps.unsqueeze(1)

#         if(self.score_type == 0):
#             final_map, final_score, img = self.compute_max_smap(score_maps, lightings)
#         elif(self.score_type == 1):
#             final_map, final_score, img = self.compute_mean_smap(score_maps, lightings)
        
#         self.image_labels.append(label.numpy())
#         self.image_preds.append(t2np(final_score))
#         self.image_list.append(t2np(img))
#         self.pixel_preds.append(t2np(final_map))
#         self.pixel_labels.extend(t2np(gt))

       
#         # display_mean_fusion(t2np(lightings), t2np(out), self.reconstruct_path, item)

# class Rec(Reconstruct_Method):
#     def __init__(self, args, cls_path):
#         super().__init__(args, cls_path)
    
#     def predict(self, item, lightings, _, gt, label):
#         lightings = lightings.squeeze().to(self.device)
#         # print(lightings.min())
#         # print(lightings.max())
    
#         out = self.rgb_model(lightings)
#         out = (out + 1) / 2
#         # out = torch.clip(out, min=lightings.min(), max=lightings.max())
#         # print(out.min())
#         # print(out.max())
#         if False:
#             # img_avg = torch.nn.AvgPool2d(3, stride=1, padding=1)
#             # imgnet_in = img_avg(lightings)
#             # imgnet_out = img_avg(out)
    
#             in_feat = self.get_pretrained_feature(lightings)
#             repaired_feat = self.get_pretrained_feature(out)
#             final_map, final_score = self.compute_feature_dist(in_feat, repaired_feat)
#             out_avg = self.average(out)
#             lightings_avg = self.average(lightings)
#             loss = self.criteria(lightings_avg, out_avg)
#         else:
#             img_avg = torch.nn.AvgPool2d(3, stride=1, padding=1)
#             out = img_avg(out)
#             lightings = img_avg(lightings)
#             loss = self.criteria(lightings, out)
#             score_maps = torch.sum(torch.abs(lightings - out), dim=1)
#             score_maps = score_maps.unsqueeze(1)
#             final_score = self.compute_score(score_maps)
#             if(self.score_type == 0):
#                 final_map, _, img = self.compute_max_smap(score_maps, lightings)
#             elif(self.score_type == 1):
#                 final_map, _, img = self.compute_mean_smap(score_maps, lightings)
#             # if item % 2 == 0:
#             display_image(t2np(lightings), t2np(out), self.reconstruct_path, item)
#         self.cls_rec_loss += loss.item()
#         self.image_labels.append(label)
#         self.image_preds.append(t2np(final_score))
#         self.image_list.append(t2np(lightings[5, :, :, :]))
#         self.pixel_preds.append(t2np(final_map))
#         self.pixel_labels.extend(t2np(gt))
