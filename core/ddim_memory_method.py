import torch

from core.base import DDIM_Method
from utils.utils import t2np
from utils.visualize_util import display_one_img, display_image
from core.models.controllora import  ControlLoRAModel
from torch.optim.adam import Adam
import torch.nn.functional as nnf

from tqdm import tqdm
from utils.ptp_utils import *

def nxn_cos_sim(A, B, dim=1):
    a_norm = nnf.normalize(A, p=2, dim=dim)
    b_norm = nnf.normalize(B, p=2, dim=dim)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

class Memory_Method(DDIM_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.f_coreset = 1
        self.coreset_eps = 0.9
        self.n_reweight = 3
        self.memory_T = args.memory_T
        self.memory_t = args.memory_t
        self.test_T = args.test_T
        self.test_t = args.test_t
        self.dist_fun = args.dist_function
    
    def get_score(self, min_val, feature_map_dims):
        s_star = torch.max(min_val)
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')
        s_map = self.blur(s_map.to('cpu'))
        return s_star.to('cpu'), s_map
            
    def compute_s_s_map(self, patch, patch_lib, feature_map_dims, p=2):
        patch_lib = patch_lib.to(self.device)
        if self.dist_fun == 'l2_dist':
            dist = torch.cdist(patch, patch_lib, p=p)
        elif self.dist_fun == 'cosine':
            dist = 1.0 - nxn_cos_sim(patch, patch_lib)
        min_val, _ = torch.min(dist, dim=1)
        return self.get_score(min_val, feature_map_dims)
    
    def compute_align_map(self, patch, patch_lib, feature_map_dims, k=2, p=2):
        patch_lib = patch_lib.to(self.device)
        if self.dist_fun == 'l2_dist':
            dist = torch.cdist(patch, patch_lib, p=p)
        elif self.dist_fun == 'cosine':
            dist = 1.0 - nxn_cos_sim(patch, patch_lib)
        min_val, min_idx = torch.topk(dist, k=k, largest=False)
        min_val = min_val[:, 1]
        return self.get_score(min_val, feature_map_dims)
    
    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0)
        
        if self.nmap_patch_lib:
            self.nmap_patch_lib = torch.cat(self.nmap_patch_lib, 0)
        
        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_lib = self.patch_lib[self.coreset_idx]

    def get_coreset_idx_randomp(self, z_lib, n=1000, eps=0.90, float16=True, force_cpu=False):
        """Returns n coreset idx for given z_lib.
        Performance on AMD3700, 32GB RAM, RTX3080 (10GB):
        CPU: 40-60 it/s, GPU: 500+ it/s (float32), 1500+ it/s (float16)
        Args:
            z_lib:      (n, d) tensor of patches.
            n:          Number of patches to select.
            eps:        Agression of the sparse random projection.
            float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
            force_cpu:  Force cpu, useful in case of GPU OOM.
        Returns:
            coreset indices
        """

        print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
        try:
            transformer = random_projection.SparseRandomProjection(eps=eps)
            z_lib = torch.tensor(transformer.fit_transform(z_lib))
            print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
        except ValueError:
            print("   Error: could not project vectors. Please increase `eps`.")

        select_idx = 0
        last_item = z_lib[select_idx:select_idx + 1]
        coreset_idx = [torch.tensor(select_idx)]
        min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
        # The line below is not faster than linalg.norm, although i'm keeping it in for
        # future reference.
        # min_distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True)

        if float16:
            last_item = last_item.half()
            z_lib = z_lib.half()
            min_distances = min_distances.half()
        if torch.cuda.is_available() and not force_cpu:
            last_item = last_item.to("cuda")
            z_lib = z_lib.to("cuda")
            min_distances = min_distances.to("cuda")

        for _ in tqdm(range(n - 1)):
            distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)  # broadcasting step
            min_distances = torch.minimum(distances, min_distances)  # iterative step
            select_idx = torch.argmax(min_distances)  # selection step

            # bookkeeping
            last_item = z_lib[select_idx:select_idx + 1]
            min_distances[select_idx] = 0
            coreset_idx.append(select_idx.to("cpu"))
        return torch.stack(coreset_idx)

    @torch.no_grad()
    def ddim_loop(self, latents, text_emb, target_timestep):
        latents = latents.clone().detach()
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        
        for t in reversed(self.timesteps_list):
            timestep = torch.tensor(t.item(), device=self.device)
            noise_pred = self.unet(latents, timestep, text_emb)['sample']
            latents = self.next_step(noise_pred, timestep, latents)
            if t == target_timestep:
                return latents, t

    @torch.no_grad() 
    def get_unet_f(self, latents, text_emb, start_t, end_t, layer=3):
        """
        start_t: Get noise latent at this timestep by using DDIM Inversion.
        end_t: Sampling noisy latent util this timestep.
        """
        assert (start_t >= end_t) and (start_t in self.timesteps_list) and (end_t in self.timesteps_list)
        noisy_latents, _  = self.ddim_loop(latents, text_emb, start_t)
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        timesteps_list = [t for t in self.timesteps_list if t <= start_t and t >= end_t]
        # print(timesteps_list)
        for t in timesteps_list:
            timestep = torch.tensor(t.item(), device=self.device)
            pred = self.unet(noisy_latents, timestep, text_emb)
            noisy_latents = self.noise_scheduler.step(pred['sample'], t.item(), noisy_latents)["prev_sample"]    
        return pred['up_ft'][layer]


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
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        latents = self.image2latents(lightings)
        bsz = latents.shape[0]
        text_emb = self.get_text_embedding(text_prompt, bsz)
        
        unet_f = self.get_unet_f(latents, text_emb, self.memory_T, self.memory_t)
        
        B, C, H, W = unet_f.shape
        
        train_unet_f = torch.mean(unet_f.view(-1, 6, C, H, W), dim=1)
        train_unet_f = train_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        self.patch_lib.append(train_unet_f.cpu())
    
    def predict(self, i, lightings, nmap, text_prompt, gt, label):
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        with torch.no_grad():
            latents = self.image2latents(lightings)
            text_emb = self.get_text_embedding(text_prompt, 6)
            unet_f = self.get_unet_f(latents, text_emb, self.test_T, self.test_t)
            # ddim_latents, timestep = self.ddim_loop(latents, text_embeddings, False)
            # unet_f = self.get_unet_latent(ddim_latents, timestep, text_embeddings)
            # unet_f = self.diffusion_loop(ddim_latents, text_embeddings)
        
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

class DDIMInvNmap_Memory(Memory_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)

    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):
        nmap = nmap.to(self.device)
        latents = self.image2latents(nmap)
        bsz = latents.shape[0]
        text_emb = self.get_text_embedding(text_prompt, bsz)
        
        unet_f = self.get_unet_f(latents, text_emb, self.memory_T, self.memory_t)
        B, C, H, W = unet_f.shape
        train_unet_f = unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        self.patch_lib.append(train_unet_f.cpu())
    
    def predict(self, i, lightings, nmap, text_prompt, gt, label):
        nmap = nmap.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        with torch.no_grad():
            latents = self.image2latents(nmap)
            text_emb = self.get_text_embedding(text_prompt, 1)
            unet_f = self.get_unet_f(latents, text_emb, self.test_T, self.test_t)
        B, C, H, W = unet_f.shape
        test_unet_f = unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        s, smap = self.compute_s_s_map(test_unet_f, self.patch_lib, unet_f.shape[-2:])
        img = lightings[5, :, :, :]
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(smap))
        self.pixel_labels.extend(t2np(gt))

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
        rgb_unet_f = self.get_unet_f(rgb_latents, rgb_text_embs, self.memory_T, self.memory_t)
        B, C, H, W = rgb_unet_f.shape
        train_rgb_unet_f = torch.mean(rgb_unet_f.view(-1, 6, C, H, W), dim=1)
        train_rgb_unet_f = train_rgb_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        self.patch_lib.append(train_rgb_unet_f.cpu())
        
        # normal  map
        nmap = nmap.to(self.device)
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_unet_f(nmap_latents, nmap_text_embs, self.memory_T, self.memory_t)
        B, C, H, W = nmap_unet_f.shape
        train_nmap_unet_f = nmap_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        self.nmap_patch_lib.append(train_nmap_unet_f.cpu())

    def predict_align_data(self, lightings, nmap, text_prompt):
        text_emb = self.get_text_embedding(text_prompt, 1)
        # rgb
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_unet_f(rgb_latents, rgb_text_embs, self.test_T, self.test_t)
        B, C, H, W = rgb_unet_f.shape
        test_rgb_unet_f = torch.mean(rgb_unet_f.view(-1, 6, C, H, W), dim=1)
        test_rgb_unet_f = test_rgb_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        rgb_s, rgb_smap = self.compute_align_map(test_rgb_unet_f, self.patch_lib, rgb_unet_f.shape[-2:])
        
        # nromal map
        nmap = nmap.to(self.device)
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_unet_f(nmap_latents, nmap_text_embs, self.test_T, self.test_t)
        B, C, H, W = nmap_unet_f.shape
        test_nmap_unet_f = nmap_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        nmap_s, nmap_smap = self.compute_align_map(test_nmap_unet_f, self.nmap_patch_lib, nmap_unet_f.shape[-2:])

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
        rgb_unet_f = self.get_unet_f(rgb_latents, rgb_text_embs, self.test_T, self.test_t)
        B, C, H, W = rgb_unet_f.shape
        test_rgb_unet_f = torch.mean(rgb_unet_f.view(-1, 6, C, H, W), dim=1)
        test_rgb_unet_f = test_rgb_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        rgb_s, rgb_smap = self.compute_s_s_map(test_rgb_unet_f, self.patch_lib, rgb_unet_f.shape[-2:])
        
        # normal map
        nmap = nmap.to(self.device)
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_unet_f(nmap_latents, nmap_text_embs, self.test_T, self.test_t)
        B, C, H, W = nmap_unet_f.shape
        test_nmap_unet_f = nmap_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        nmap_s, nmap_smap = self.compute_s_s_map(test_nmap_unet_f, self.nmap_patch_lib, nmap_unet_f.shape[-2:])
        
        # s, pixel_map = self.compute_cosine_score(test_rgb_unet_f, test_nmap_unet_f, nmap_unet_f.shape[-2:])
                # nmap distribution
        
        # print("rgb_s:", rgb_s * self.weight + self.bias)
        # print("nmap_s:", nmap_s)
        pixel_map = (rgb_smap * self.weight + self.bias) + nmap_smap
        s = (rgb_s * self.weight + self.bias) + nmap_s
        
        ### Record Score ###
        img = lightings[5, :, :, :]
        self.image_list.append(t2np(img))
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.rgb_image_preds.append(rgb_s.numpy())
        self.nmap_image_preds.append(nmap_s.numpy())
        
        self.pixel_labels.append(t2np(gt))
        self.pixel_preds.append(t2np(pixel_map))
        self.rgb_pixel_preds.append(t2np(rgb_smap))
        self.nmap_pixel_preds.append(t2np(nmap_smap))

class DDIMInvUnified_MultiMemory(Memory_Method):        
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.weight = 1
        self.bias = 0
        self.multi_patch_lib = []
        
    def compute_mul_s_s_map(self, multi_patch, feature_map_dims, mode="testing", k=2, p=2):
        patch_lib = self.patch_lib.to(self.device)
        mean_patch = torch.mean(multi_patch, dim=1)
        if self.dist_fun == 'l2_dist':
            dist = torch.cdist(mean_patch, patch_lib, p=p)
        elif self.dist_fun == 'cosine':
            dist = 1.0 - nxn_cos_sim(mean_patch, patch_lib)
        
        if mode == "testing":
            min_val, min_idx = torch.min(dist, dim=1)
        else:
            min_val, min_idx = torch.topk(dist, k=k, largest=False)
            min_val = min_val[:, 1]
            min_idx = min_idx[:, 1]
        mem_multi_patch = self.multi_patch_lib[min_idx]
        mem_multi_patch = mem_multi_patch.to(self.device)
        six_smap = self.pdist(mem_multi_patch, multi_patch)
        s_map, _ = torch.max(six_smap, dim=1)
        return self.get_score(s_map, feature_map_dims)
          
    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0)
        
        if self.nmap_patch_lib:
            self.nmap_patch_lib = torch.cat(self.nmap_patch_lib, 0)
            
        if self.multi_patch_lib:
            self.multi_patch_lib = torch.cat(self.multi_patch_lib, 0)
            
        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_lib = self.patch_lib[self.coreset_idx]        
        
    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):
        text_emb = self.get_text_embedding(text_prompt, 1)
        
        # rgb
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_unet_f(rgb_latents, rgb_text_embs, self.memory_T, self.memory_t)
        B, C, H, W = rgb_unet_f.shape

        multi_rgb_unet_f = rgb_unet_f.view(-1, 6, C, H, W)
    
        
        multi_rgb_unet_f = multi_rgb_unet_f.permute(0, 3, 4, 1, 2).reshape((B // 6) * H * W, 6, C)
        mean_rgb_unet_f = torch.mean(multi_rgb_unet_f, dim=1)
        self.multi_patch_lib.append(multi_rgb_unet_f.cpu())
        self.patch_lib.append(mean_rgb_unet_f.cpu())
        
        # normal  map
        nmap = nmap.to(self.device)
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_unet_f(nmap_latents, nmap_text_embs, self.memory_T, self.memory_t)
        B, C, H, W = nmap_unet_f.shape
        train_nmap_unet_f = nmap_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        self.nmap_patch_lib.append(train_nmap_unet_f.cpu())

    def predict_align_data(self, lightings, nmap, text_prompt):
        text_emb = self.get_text_embedding(text_prompt, 1)
        # rgb
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_unet_f(rgb_latents, rgb_text_embs, self.test_T, self.test_t)
        B, C, H, W = rgb_unet_f.shape
        
        multi_rgb_unet_f = rgb_unet_f.view(-1, 6, C, H, W)
        
        multi_rgb_unet_f = multi_rgb_unet_f.permute(0, 3, 4, 1, 2).reshape((B // 6) * H * W, 6, C)
        rgb_s, rgb_smap = self.compute_mul_s_s_map(multi_rgb_unet_f, rgb_unet_f.shape[-2:], mode="alginment")
        
        # nromal map
        nmap = nmap.to(self.device)
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_unet_f(nmap_latents, nmap_text_embs, self.test_T, self.test_t)
        B, C, H, W = nmap_unet_f.shape
        test_nmap_unet_f = nmap_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        nmap_s, nmap_smap = self.compute_align_map(test_nmap_unet_f, self.nmap_patch_lib, nmap_unet_f.shape[-2:])

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
        rgb_unet_f = self.get_unet_f(rgb_latents, rgb_text_embs, self.test_T, self.test_t)
        B, C, H, W = rgb_unet_f.shape
        multi_rgb_unet_f = rgb_unet_f.view(-1, 6, C, H, W)
        multi_rgb_unet_f = multi_rgb_unet_f.permute(0, 3, 4, 1, 2).reshape((B // 6) * H * W, 6, C)
        rgb_s, rgb_smap = self.compute_mul_s_s_map(multi_rgb_unet_f, rgb_unet_f.shape[-2:], mode="testing")
        
        # normal map
        nmap = nmap.to(self.device)
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_unet_f(nmap_latents, nmap_text_embs, self.test_T, self.test_t)
        B, C, H, W = nmap_unet_f.shape
        test_nmap_unet_f = nmap_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        nmap_s, nmap_smap = self.compute_s_s_map(test_nmap_unet_f, self.nmap_patch_lib, nmap_unet_f.shape[-2:])
        
        pixel_map = rgb_smap * self.weight + self.bias + nmap_smap
        s = (rgb_s * self.weight + self.bias) + nmap_s
        img = lightings[5, :, :, :]
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.image_list.append(t2np(img))
        self.rgb_image_preds.append(rgb_s.numpy())
        self.nmap_image_preds.append(nmap_s.numpy())
        
        self.pixel_preds.append(t2np(pixel_map))
        self.pixel_labels.extend(t2np(gt))
        self.rgb_pixel_preds.extend(rgb_smap.flatten().numpy())
        self.nmap_pixel_preds.extend(nmap_smap.flatten().numpy())
        
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

        self.weight = 1
        self.bias = 0
        
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
    def ddim_loop(self, latents, condition_map, text_emb, target_timestep):
        latents = latents.clone().detach()
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        for t in reversed(self.timesteps_list):
            noise_pred = self.controlnet(latents, condition_map, t, text_emb)['sample']
            latents = self.next_step(noise_pred, t, latents)
            if t == target_timestep:
                return latents, t
    
    @torch.no_grad() 
    def get_unet_f(self, latents, condition_map, text_emb, start_t, end_t):
        """
        start_t: Get noise latent at this timestep by using DDIM Inversion.
        end_t: Sampling noisy latent util this timestep.
        """
        assert (start_t >= end_t) and (start_t in self.timesteps_list) and (end_t in self.timesteps_list)
        noisy_latents, _  = self.ddim_loop(latents, condition_map, text_emb, start_t)
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        timesteps_list = [t for t in self.timesteps_list if t <= start_t and t >= end_t]
        for t in timesteps_list:
            timestep = torch.tensor(t.item(), device=self.device)
            pred = self.controlnet(noisy_latents, condition_map, timestep, text_emb)
            noisy_latents = self.noise_scheduler.step(pred['sample'], t.item(), noisy_latents)["prev_sample"]    
        return pred['up_ft'][3]
   
    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):

        text_emb = self.get_text_embedding(text_prompt, 1)
        lightings = lightings.to(self.device)
        nmap = nmap.to(self.device)

        single_lightings = lightings[:, 5, :, :, :] # [bs, 3, 256, 256]
        repeat_nmaps = nmap.repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]

        # rgb
        lightings = lightings.view(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_unet_f(rgb_latents, repeat_nmaps, rgb_text_embs, self.memory_T, self.memory_t)
        B, C, H, W = rgb_unet_f.shape
        train_rgb_unet_f = torch.mean(rgb_unet_f.view(-1, 6, C, H, W), dim=1)
        train_rgb_unet_f = train_rgb_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        self.patch_lib.append(train_rgb_unet_f.cpu())

        # normal  map
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_unet_f(nmap_latents, single_lightings, nmap_text_embs, self.memory_T, self.memory_t)
        B, C, H, W = nmap_unet_f.shape
        train_nmap_unet_f = nmap_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        self.nmap_patch_lib.append(train_nmap_unet_f.cpu())

    def predict_align_data(self, lightings, nmap, text_prompt):
        
        text_emb = self.get_text_embedding(text_prompt, 1)
        lightings = lightings.to(self.device)
        nmap = nmap.to(self.device)

        single_lightings = lightings[:, 5, :, :, :] # [bs, 3, 256, 256]
        repeat_nmaps = nmap.repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]
        
        # rgb
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_unet_f(rgb_latents, repeat_nmaps, rgb_text_embs, self.test_T, self.test_t)
        B, C, H, W = rgb_unet_f.shape
        test_rgb_unet_f = torch.mean(rgb_unet_f.view(-1, 6, C, H, W), dim=1)
        test_rgb_unet_f = test_rgb_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        rgb_s, rgb_smap = self.compute_align_map(test_rgb_unet_f, self.patch_lib, rgb_unet_f.shape[-2:])
        
        # nromal map
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_unet_f(nmap_latents, single_lightings, nmap_text_embs, self.test_T, self.test_t)
        B, C, H, W = nmap_unet_f.shape
        test_nmap_unet_f = nmap_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        nmap_s, nmap_smap = self.compute_align_map(test_nmap_unet_f, self.nmap_patch_lib, nmap_unet_f.shape[-2:])

        # image_level
        self.nmap_image_preds.append(nmap_s.numpy())
        self.rgb_image_preds.append(rgb_s.numpy())
        # pixel_level
        self.rgb_pixel_preds.extend(rgb_smap.flatten().numpy())
        self.nmap_pixel_preds.extend(nmap_smap.flatten().numpy())    

    def predict(self, i, lightings, nmap, text_prompt, gt, label):
        
        text_emb = self.get_text_embedding(text_prompt, 1)
        lightings = lightings.to(self.device)
        nmap = nmap.to(self.device)

        single_lightings = lightings[:, 5, :, :, :] # [bs, 3, 256, 256]
        repeat_nmaps = nmap.repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]
        
        # rgb
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_unet_f(rgb_latents, repeat_nmaps, rgb_text_embs, self.test_T, self.test_t)
        B, C, H, W = rgb_unet_f.shape
        test_rgb_unet_f = torch.mean(rgb_unet_f.view(-1, 6, C, H, W), dim=1)
        test_rgb_unet_f = test_rgb_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        rgb_s, rgb_smap = self.compute_s_s_map(test_rgb_unet_f, self.patch_lib, rgb_unet_f.shape[-2:])
        
        # normal map
        nmap = nmap.to(self.device)
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_unet_f(nmap_latents, single_lightings, nmap_text_embs, self.test_T, self.test_t)
        B, C, H, W = nmap_unet_f.shape
        test_nmap_unet_f = nmap_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        nmap_s, nmap_smap = self.compute_s_s_map(test_nmap_unet_f, self.nmap_patch_lib, nmap_unet_f.shape[-2:])


        ### Combine RGB and Nmap score map ###
        #s = torch.maximum(rgb_s, nmap_s)
        #s = rgb_s * nmap_s
        #s = (rgb_s * self.weight + self.bias) + nmap_s
        s = torch.maximum((rgb_s * self.weight + self.bias), nmap_s)

        #smap = (rgb_smap * self.weight + self.bias) + nmap_smap
        smap = torch.maximum((rgb_smap * self.weight + self.bias), nmap_smap)
        #new_rgb_map = rgb_smap * self.weight + self.bias
        #new_rgb_map = torch.clip(new_rgb_map, min=0, max=new_rgb_map.max())
        #smap = torch.maximum(new_rgb_map, nmap_smap)
        #smap = new_rgb_map + nmap_smap
        #s = rgb_s * self.weight + nmap_s

        img = lightings[5, :, :, :]
        self.image_list.append(t2np(img))
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.rgb_image_preds.append(rgb_s.numpy())
        self.nmap_image_preds.append(nmap_s.numpy())
        
        self.pixel_labels.append(t2np(gt))
        self.pixel_preds.append(t2np(smap))
        self.rgb_pixel_preds.append(t2np(rgb_smap))
        self.nmap_pixel_preds.append(t2np(nmap_smap))
        
        
# class DDIMInvRGBNmap_Memory(Memory_Method):
#     def __init__(self, args, cls_path):
#         super().__init__(args, cls_path)

#     @torch.no_grad()
#     def ddim_loop(self, latents, nmap, text_emb, target_timestep):
#         latents = latents.clone().detach()
#         self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
#         _, _, h, w = latents.shape
#         for t in reversed(self.timesteps_list):
#             timestep = torch.tensor(t.item(), device=self.device)
#             concat_in = torch.cat((latents, nnf.interpolate(nmap, (h, w))), 1)
#             noise_pred = self.unet(concat_in, timestep, text_emb)['sample']
#             latents = self.next_step(noise_pred, timestep, latents)
#             if t == target_timestep:
#                 return latents, t
    
#     @torch.no_grad() 
#     def get_unet_f(self, latents, nmap, text_emb, start_t, end_t):
#         """
#         start_t: Get noise latent at this timestep by using DDIM Inversion.
#         end_t: Sampling noisy latent util this timestep.
#         """
#         assert (start_t >= end_t) and (start_t in self.timesteps_list) and (end_t in self.timesteps_list)
#         _, _, h, w = latents.shape
#         noisy_latents, _  = self.ddim_loop(latents, nmap, text_emb, start_t)
#         self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
#         timesteps_list = [t for t in self.timesteps_list if t <= start_t and t >= end_t]
#         # print(timesteps_list)
#         for t in timesteps_list:
#             timestep = torch.tensor(t.item(), device=self.device)
#             concat_in = torch.cat((noisy_latents, nnf.interpolate(nmap, (h, w))), 1)
#             pred = self.unet(concat_in, timestep, text_emb)
#             noisy_latents = self.noise_scheduler.step(pred['sample'], t.item(), noisy_latents)["prev_sample"]    
#         return pred['up_ft'][3]
    
#     def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):
#         lightings = lightings.to(self.device)
#         nmaps = nmap.to(self.device).repeat_interleave(6, dim=0)
#         lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
#         latents = self.image2latents(lightings)
#         bsz = latents.shape[0]
#         text_emb = self.get_text_embedding(text_prompt, bsz)
        
#         unet_f = self.get_unet_f(latents, nmaps, text_emb, self.memory_T, self.memory_t)
        
#         B, C, H, W = unet_f.shape
        
#         train_unet_f = torch.mean(unet_f.view(-1, 6, C, H, W), dim=1)
#         train_unet_f = train_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
#         self.patch_lib.append(train_unet_f.cpu())
    
#     def predict(self, i, lightings, nmap, text_prompt, gt, label):
#         lightings = lightings.to(self.device)
#         lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
#         nmaps = nmap.to(self.device).repeat_interleave(6, dim=0)
#         with torch.no_grad():
#             latents = self.image2latents(lightings)
#             text_emb = self.get_text_embedding(text_prompt, 6)
#             unet_f = self.get_unet_f(latents, nmaps, text_emb, self.test_T, self.test_t)
#             # ddim_latents, timestep = self.ddim_loop(latents, text_embeddings, False)
#             # unet_f = self.get_unet_latent(ddim_latents, timestep, text_embeddings)
#             # unet_f = self.diffusion_loop(ddim_latents, text_embeddings)
        
#         B, C, H, W = unet_f.shape
        
#         test_unet_f = torch.mean(unet_f.view(-1, 6, C, H, W), dim=1)
#         test_unet_f = test_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T

#         s, smap = self.compute_s_s_map(test_unet_f, unet_f.shape[-2:])
#         img = lightings[5, :, :, :]
#         self.image_labels.append(label.numpy())
#         self.image_preds.append(s.numpy())
#         self.image_list.append(t2np(img))
#         self.pixel_preds.append(t2np(smap))
#         self.pixel_labels.extend(t2np(gt))