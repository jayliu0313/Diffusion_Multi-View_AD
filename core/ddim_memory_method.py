import torch
import torch.nn as nn
from torchvision import transforms
from core.base import DDIM_Method
from utils.utils import t2np, nxn_cos_sim
from core.models.controllora import  ControlLoRAModel
from torch.optim.adam import Adam
# from scipy.stats import wasserstein_distance
# from scipy.spatial.distance import cdist
from utils.ptp_utils import *
from geomloss import SamplesLoss
from utils.visualize_util import display_one_img
# from sinkhorn import sinkhorn
# # from pyemd import emd
# from dist_matrix.cuda_dist_matrix_full import dist_matrix as gpu_dist_matrix
# import ot
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import ot

import os
import os.path as osp
np.seterr(divide='ignore', invalid='ignore')

class Memory_Method(DDIM_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.f_coreset = 1
        self.coreset_eps = 0.9
        self.n_reweight = 3
        self.target_timestep = max(args.noise_intensity)
        # print(self.target_timestep)
        self.patch_lib = []
        self.nmap_patch_lib = []
        
        
    def compute_s_s_map(self, patch, patch_lib, feature_map_dims, p=2, k=1):
        _, _, C, _, _ = patch.shape
        target_patch_lib = patch_lib[-1].permute(1, 0, 2, 3).reshape(C, -1).T.to(self.device)
        target_patch = patch[-1].permute(1, 0, 2, 3).reshape(C, -1).T.to(self.device)
       
        # target_patch_lib = patch_lib[-1].to(self.device)
        # target_patch = patch[-1].to(self.device)
        dist = torch.cdist(target_patch, target_patch_lib, p=p)
        smap, min_idx = torch.topk(dist, k=k, largest=False)
        smap = smap[:, -1]
        min_idx = min_idx[:, -1]
        min_idx = min_idx.to('cpu')
        if len(patch_lib) > 1:
            mul_smap = self.pdist(patch.to(self.device), patch_lib[:, min_idx, :].to(self.device))
            smap, _ = torch.max(mul_smap, dim=0)
        #s_star = torch.max(smap)
        topk_value, _ = torch.topk(smap, k=self.topk)
        s_star = torch.mean(topk_value)
        if self.reweight:
            s_idx = torch.argmax(smap)
            m_test = target_patch[s_idx].unsqueeze(0)  # anomalous patch
            m_star = target_patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, target_patch_lib)  # find knn to m_star pt.1
            _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

            m_star_knn = torch.linalg.norm(m_test - target_patch_lib[nn_idx[0, 1:]], dim=1)
            D = torch.sqrt(torch.tensor(target_patch.shape[1]))
            w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
            s_star = w * s_star
        smap = smap.view(1, 1, *feature_map_dims)
        smap = torch.nn.functional.interpolate(smap, size=(self.image_size, self.image_size), mode='bilinear')
        smap = self.blur(smap.to('cpu'))

        return s_star.to('cpu'), smap

    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib , 1)
        if self.nmap_patch_lib:
            self.nmap_patch_lib = torch.cat(self.nmap_patch_lib , 1)

    @torch.no_grad()
    def ddim_loop(self, latents, text_emb):
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        latent_list = []
        t_list = []
        for t in reversed(self.timesteps_list):
            noise_pred = self.unet(latents.to(self.device), t, text_emb)['sample']
            latents = self.next_step(noise_pred, t, latents)
            if t in self.mul_timesteps:
                latent_list.append(latents.to('cpu'))
                t_list.append(t)
        return latent_list, t_list
    
    @torch.no_grad()
    def get_feature_layers(self, latent, t, text_emb):
        pred_f = self.unet(latent.to(self.device), t, text_emb)['up_ft']
        resized_f = []
        for i in range(len(pred_f)):
            if i in self.feature_layers:
                largest_fmap_size = pred_f[self.feature_layers[-1]].shape[-2:]
                resized_f.append(torch.nn.functional.interpolate(pred_f[i], largest_fmap_size, mode='bicubic').to('cpu'))
        features = torch.cat(resized_f, dim=1)
        return features
        
    @torch.no_grad() 
    def get_unet_f(self, latents, text_emb, islighting=True):
        latent_list, t_list = self.ddim_loop(latents, text_emb)
        unetf_list = []
        for i, t in enumerate(t_list):

            unet_f = self.get_feature_layers(latent_list[i], t, text_emb)
            B, C, H, W = unet_f.shape
            if islighting and self.data_type=="eyecandies":
                unet_f = torch.mean(unet_f.view(-1, 6, C, H, W), dim=1)
            # unet_f = unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
            unetf_list.append(unet_f.to('cpu'))

        unet_fs = torch.stack(unetf_list)
        return unet_fs

class DDIM_Memory(Memory_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        
    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        latents = self.image2latents(lightings)
        bsz = latents.shape[0]
        text_embeddings = self.get_text_embedding(text_prompt, bsz)
        _, timesteps, noisy_latents = self.forward_process_with_T(latents, self.timesteps_list[-1])
        # print(timesteps)
        model_output = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings,
        )
        
        unet_f = model_output['up_ft'][3]
        B, C, H, W = unet_f.shape
        
        train_unet_f = torch.mean(unet_f.view(-1, 6, C, H, W), dim=1)
        train_unet_f = train_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        self.patch_lib.append(train_unet_f.cpu())
    
    def predict(self, i, lightings, nmap, text_prompt, gt, label):
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
    
        latents = self.image2latents(lightings)
        cond_embeddings = self.get_text_embedding(text_prompt, 6)

    
        _, timesteps, noisy_latents = self.forward_process_with_T(latents, self.timesteps_list[-1])
        # print(timesteps)
        model_output = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=cond_embeddings,
        )
        
        unet_f = model_output['up_ft'][3]
        B, C, H, W = unet_f.shape
        
        test_unet_f = torch.mean(unet_f.view(-1, 6, C, H, W), dim=1)
        test_unet_f = test_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T

        s, smap = self.compute_s_s_map(test_unet_f, self.patch_lib, unet_f.shape[-2:])
        img = lightings[5, :, :, :]
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(smap))
        self.pixel_labels.extend(t2np(gt))

class DDIMInvRGB_Memory(Memory_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)

    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):
        lightings = lightings.to(self.device)
        #lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        latents = self.image2latents(lightings)
        bsz = latents.shape[0]
        text_emb = self.get_text_embedding(text_prompt, bsz)
        
        unet_f = self.get_unet_f(latents, text_emb, islighting=False)
        self.patch_lib.append(unet_f.cpu())
    
    def predict(self, i, lightings, nmap, text_prompt, gt, label):
        lightings = lightings.to(self.device)
        #lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        
        latents = self.image2latents(lightings)
        text_emb = self.get_text_embedding(text_prompt, 1)
        unet_f = self.get_unet_f(latents, text_emb, islighting=False)
        
        s, smap = self.compute_s_s_map(unet_f, self.patch_lib, latents.shape[-2:])
        img = lightings[0]
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(smap))
        self.pixel_labels.extend(t2np(gt))

class DDIMInvNmap_Memory(Memory_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)

    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):
        nmap = nmap.to(self.device)
        latents = self.image2latents(nmap)
        bsz = latents.shape[0]
        text_emb = self.get_text_embedding(text_prompt, bsz)
        
        unet_f = self.get_unet_f(latents, text_emb, islighting=False)
        self.patch_lib.append(unet_f.cpu())


    def predict(self, i, lightings, nmap, text_prompt, gt, label):
        nmap = nmap.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        latents = self.image2latents(nmap)
        text_emb = self.get_text_embedding(text_prompt, 1)
        unet_f = self.get_unet_f(latents, text_emb, islighting=False)
        s, smap = self.compute_s_s_map(unet_f, self.patch_lib, latents.shape[-2:])
        img = lightings[5, :, :, :]
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(smap))
        self.pixel_labels.extend(t2np(gt))
        # print(img)
        
class DDIMInvUnified_Memory(Memory_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.weight = 1
        self.bias = 0
    
    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):
        text_emb = self.get_text_embedding(text_prompt, 1)
        
        # rgb
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_unet_f(rgb_latents, rgb_text_embs, islighting=True)
        self.patch_lib.append(rgb_unet_f.cpu())
        
        # normal  map
        nmap = nmap.to(self.device)
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_unet_f(nmap_latents, nmap_text_embs, islighting=False)
        self.nmap_patch_lib.append(nmap_unet_f.cpu())

    def predict_align_data(self, lightings, nmap, text_prompt):
        text_emb = self.get_text_embedding(text_prompt, 1)
        # rgb
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_unet_f(rgb_latents, rgb_text_embs)
        rgb_s, rgb_smap = self.compute_s_s_map(rgb_unet_f, self.patch_lib, rgb_latents.shape[-2:], k=2)
        
        # nromal map
        nmap = nmap.to(self.device)
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_unet_f(nmap_latents, nmap_text_embs)
        nmap_s, nmap_smap = self.compute_s_s_map(nmap_unet_f, self.nmap_patch_lib, nmap_latents.shape[-2:], k=2)

        # image_level
        self.nmap_image_preds.append(nmap_s.numpy())
        self.rgb_image_preds.append(rgb_s.numpy())
        # pixel_level
        self.rgb_pixel_preds.extend(rgb_smap.flatten().numpy())
        self.nmap_pixel_preds.extend(nmap_smap.flatten().numpy())

    @ torch.no_grad()
    def predict(self, i, lightings, nmap, text_prompt, gt, label):
        text_emb = self.get_text_embedding(text_prompt, 1)
        # rgb
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_unet_f(rgb_latents, rgb_text_embs, islighting=True)
        rgb_s, rgb_smap = self.compute_s_s_map(rgb_unet_f, self.patch_lib, rgb_latents.shape[-2:])
        
        # normal map
        nmap = nmap.to(self.device)
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_unet_f(nmap_latents, nmap_text_embs, islighting=False)
        nmap_s, nmap_smap = self.compute_s_s_map(nmap_unet_f, self.nmap_patch_lib, nmap_latents.shape[-2:])

        pixel_map = rgb_smap + nmap_smap
        s = rgb_s + nmap_s
        
        ### Record Score ###
        img = lightings[-1, :, :, :]
        self.image_list.append(t2np(img))
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.rgb_image_preds.append(rgb_s.numpy())
        self.nmap_image_preds.append(nmap_s.numpy())
        
        self.pixel_labels.append(t2np(gt))
        self.pixel_preds.append(t2np(pixel_map))
        self.rgb_pixel_preds.append(t2np(rgb_smap))
        self.nmap_pixel_preds.append(t2np(nmap_smap))

class ControlNet_DDIMInv_Memory(Memory_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)

        # Setting ControlNet Model 
        print("Loading ControlNet")
        self.controllora = ControlLoRAModel.from_unet(self.unet, lora_linear_rank=args.controllora_linear_rank, lora_conv2d_rank=args.controllora_conv2d_rank)
        self.controllora.load_state_dict(torch.load(args.load_controlnet_ckpt, map_location=self.device))
        self.controllora.tie_weights(self.unet)
        self.controllora.requires_grad_(False)
        self.controllora.eval()
        self.controllora.to(self.device)

        self.weight = args.rgb_weight
        self.bias = 0
        self.nmap_weight = args.nmap_weight
        
    @torch.no_grad()
    def get_feature_layers(self, latent, condition_map, t, text_emb):
        pred_f = self.controlnet(latent.to(self.device), condition_map, t, text_emb)['up_ft']
        features_list = []
        for layer in self.feature_layers:
            resized_f = torch.nn.functional.interpolate(pred_f[layer], size=(32, 32), mode='bicubic')
            features_list.append(resized_f.to('cpu'))
        features = torch.cat(features_list, dim=1)
        return features
    
    @torch.no_grad()    
    def controlnet(self, noisy_latents, condition_map, timestep, text_emb):
        down_block_res_samples, mid_block_res_sample = self.controllora(
            noisy_latents, timestep,
            encoder_hidden_states=text_emb,
            controlnet_cond=condition_map,
            guess_mode=False, return_dict=False,
        )
        model_output = self.unet(
            noisy_latents, timestep,
            encoder_hidden_states=text_emb,
            down_block_additional_residuals=[sample for sample in down_block_res_samples],
            mid_block_additional_residual=mid_block_res_sample
        )
        return model_output
    
    @torch.no_grad()
    def controlnet_ddim_loop(self, latents, condition_map, text_emb):
        latents = latents.clone().detach()
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        latent_list = []
        t_list = []
        for t in reversed(self.timesteps_list):
            noise_pred = self.controlnet(latents.to(self.device), condition_map, t, text_emb)['sample']
            latents = self.next_step(noise_pred, t, latents)
            if t in self.mul_timesteps:
                latent_list.append(latents.to('cpu'))
                t_list.append(t)
        return latent_list, t_list
    
    @torch.no_grad() 
    def get_controlnet_f(self, latents, condition_map, text_emb, islighting=True):
        noisy_latents, t_list  = self.controlnet_ddim_loop(latents, condition_map, text_emb)
        control_fs = []
        for i, t in enumerate(t_list):
            contol_f = self.get_feature_layers(noisy_latents[i].to(self.device), condition_map, t, text_emb)
            B, C, H, W = contol_f.shape
            if islighting and self.data_type=="eyecandies":
                contol_f = torch.mean(contol_f.view(-1, 6, C, H, W), dim=1)
            control_fs.append(contol_f.to('cpu'))
        control_fs = torch.stack(control_fs)
        return control_fs
    
    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):

        text_emb = self.get_text_embedding(text_prompt, 1)
        
        lightings = lightings.to(self.device)
        
        nmap = nmap.to(self.device)

        # single_lightings = lightings[:, 5, :, :, :] # [bs, 3, 256, 256]
        if self.data_type == "eyecandies":
            nmap_repeat = nmap.repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]
        else:
            nmap_repeat = nmap
        # rgb
        lightings = lightings.view(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_controlnet_f(rgb_latents, nmap_repeat, rgb_text_embs)
        self.patch_lib.append(rgb_unet_f.cpu())

        # normal  map
        nmap_latents = self.image2latents(nmap)
        if self.data_type == "eyecandies":
            nmap_latents = nmap_latents.repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]      
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_controlnet_f(nmap_latents, lightings, nmap_text_embs)
        self.nmap_patch_lib.append(nmap_unet_f.cpu())  

    def predict(self, i, lightings, nmap, text_prompt, gt, label):
        
        text_emb = self.get_text_embedding(text_prompt, 1)
        lightings = lightings.to(self.device)
        nmap = nmap.to(self.device)

        # single_lightings = lightings[:, 5, :, :, :] # [bs, 3, 256, 256]
        if self.data_type == "eyecandies":
            nmap_repeat = nmap.repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]
        else:
            nmap_repeat = nmap
        # rgb
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_controlnet_f(rgb_latents, nmap_repeat, rgb_text_embs)
        rgb_s, rgb_smap = self.compute_s_s_map(rgb_unet_f, self.patch_lib, rgb_latents.shape[-2:])
        
        # normal map
        nmap = nmap.to(self.device)
        nmap_latents = self.image2latents(nmap)
        if self.data_type == "eyecandies":
            nmap_latents = nmap_latents.repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]      
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_controlnet_f(nmap_latents, lightings, nmap_text_embs)
        nmap_s, nmap_smap = self.compute_s_s_map(nmap_unet_f, self.nmap_patch_lib, nmap_latents.shape[-2:])

        ### Combine RGB and Nmap score map ###
        s = rgb_s * self.weight + nmap_s * self.nmap_weight
        smap = rgb_smap + nmap_smap
        # s, _ = torch.topk(smap, k=self.topk)
        # s = torch.mean(s)
        img = lightings[-1, :, :, :]
        self.image_list.append(t2np(img))
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.rgb_image_preds.append(rgb_s.numpy())
        self.nmap_image_preds.append(nmap_s.numpy())
        
        self.pixel_labels.append(t2np(gt))
        self.pixel_preds.append(t2np(smap))
        self.rgb_pixel_preds.append(t2np(rgb_smap))
        self.nmap_pixel_preds.append(t2np(nmap_smap))
