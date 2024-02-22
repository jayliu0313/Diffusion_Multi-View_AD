import torch

from core.base import DDIM_Method
from utils.utils import t2np, nxn_cos_sim
from utils.visualize_util import display_one_img, display_image
from core.models.controllora import  ControlLoRAModel
from torch.optim.adam import Adam
import torch.nn.functional as nnf

from tqdm import tqdm
from utils.ptp_utils import *

class MT_Memory_Method(DDIM_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.f_coreset = 1
        self.coreset_eps = 0.9
        self.n_reweight = 3
        self.target_timestep = args.noise_intensity
        self.mul_timesteps = args.multi_timesteps
        self.dist_fun = args.dist_function
        self.patch_lib = []
        self.nmap_patch_lib = []
          
    def compute_s_s_map(self, patch, patch_lib, feature_map_dims, p=2):
        # torch.cuda.empty_cache()
        target_patch_lib = patch_lib[-1]
        target_patch = patch[-1]
        if self.dist_fun == 'l2_dist':
            dist = torch.cdist(target_patch.to(self.device), target_patch_lib.to(self.device), p=p)
        elif self.dist_fun == 'cosine':
            dist = 1.0 - nxn_cos_sim(target_patch, target_patch_lib)
        _, min_idx = torch.min(dist, dim=1)
        mem_patches = patch_lib[:, min_idx, :]
        mul_smap = self.pdist(patch.to(self.device), mem_patches.to(self.device))
        smap, _ = torch.max(mul_smap, dim=0)
        s_star = torch.max(smap)
        smap = smap.view(1, 1, *feature_map_dims)
        smap = torch.nn.functional.interpolate(smap, size=(self.image_size, self.image_size), mode='bilinear')
        smap = self.blur(smap.to('cpu'))
        return s_star.to('cpu'), smap
        
    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib , 1)
        if self.nmap_patch_lib:
            self.nmap_patch_lib = torch.cat(self.nmap_patch_lib , 1)
        
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
    def ddim_loop(self, latents, text_emb):
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        latent_list = []
        t_list = []
        for t in reversed(self.timesteps_list):
            noise_pred = self.unet(latents, t, text_emb)['sample']
            latents = self.next_step(noise_pred, t, latents)
            if t in self.mul_timesteps:
                latent_list.append(latents.to('cpu'))
                t_list.append(t)
        return latent_list, t_list

    @torch.no_grad() 
    def get_unet_f(self, latents, text_emb, layer=3):
        latent_list, t_list = self.ddim_loop(latents, text_emb)
        unetf_list = []
        for i, t in enumerate(t_list):
            unet_f = self.unet(latent_list[i].to(self.device), t, text_emb)['up_ft'][layer]
            B, C, H, W = unet_f.shape
            unet_f = torch.mean(unet_f.view(-1, 6, C, H, W), dim=1)
            unet_f = unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
            unetf_list.append(unet_f.to('cpu'))
        unet_fs = torch.stack(unetf_list)
        return unet_fs

class DDIMInvRGB_MTMemory(MT_Memory_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)

    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        latents = self.image2latents(lightings)
        bsz = latents.shape[0]
        text_emb = self.get_text_embedding(text_prompt, bsz)
        unet_fs = self.get_unet_f(latents, text_emb)
        
        self.patch_lib.append(unet_fs.to('cpu'))
        
    
    def predict(self, i, lightings, nmap, text_prompt, gt, label):
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        img = lightings[5, :, :, :]
        lightings = lightings.to(self.device)

        latents = self.image2latents(lightings)
        text_emb = self.get_text_embedding(text_prompt, 6)
        unet_fs = self.get_unet_f(latents, text_emb)
        s, smap = self.compute_s_s_map(unet_fs, self.patch_lib, latents.shape[-2:])
        
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(smap))
        self.pixel_labels.extend(t2np(gt))
    

class ControlNet_DDIMInv_MTMemory(MT_Memory_Method):
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
    def controlnet_ddim_loop(self, latents, condition_map, text_emb):
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        latents_list = []
        t_list = []
        for t in reversed(self.timesteps_list):
            timestep = torch.tensor(t.item(), device=self.device)
    
            noise_pred = self.controlnet(latents, condition_map, timestep, text_emb)['sample']
            latents = self.next_step(noise_pred, timestep.item(), latents)
            
            if t in self.mul_timesteps:
                latents_list.append(latents.to('cpu'))
                t_list.append(t)
        return latents_list, t_list

                
    @torch.no_grad() 
    def get_controlnet_f(self, latents, condition_map, text_emb, layer=3):
        """
        start_t: Get noise latent at this timestep by using DDIM Inversion.
        end_t: Sampling noisy latent util this timestep.
        """
        noisy_latents, t_list = self.controlnet_ddim_loop(latents, condition_map, text_emb)
        controlf_list = []
        for i, t in enumerate(t_list):
            control_f = self.controlnet(noisy_latents[i], condition_map, t, text_emb)['up_ft'][layer]
            B, C, H, W = control_f.shape
            control_f = torch.mean(control_f.view(-1, 6, C, H, W), dim=1)
            control_f = control_f.permute(1, 0, 2, 3).reshape(C, -1).T
            controlf_list.append(control_f.to('cpu'))
        control_fs = torch.stack(controlf_list)
        print(control_fs.shape)    
        return control_fs

    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):
        text_emb = self.get_text_embedding(text_prompt, 1)
        lightings = lightings.to(self.device)
        nmap = nmap.to(self.device)

        single_lightings = lightings[:, 5, :, :, :] # [bs, 3, 256, 256]
        # repeat_nmaps = nmap.repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]

        # rgb
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        latents = self.image2latents(lightings)
        bsz = latents.shape[0]
        text_emb = self.get_text_embedding(text_prompt, bsz)
        unet_fs = self.get_unet_f(latents, text_emb)
        for i in range(len(unet_fs)):
            self.patch_lib[i].append(unet_fs[i].to('cpu'))

        # normal  map
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_control_f = self.get_controlnet_f(nmap_latents, single_lightings, nmap_text_embs)
        self.nmap_patch_lib.append(nmap_control_f.cpu())

    def predict_align_data(self, lightings, nmap, text_prompt):
        text_emb = self.get_text_embedding(text_prompt, 1)
        lightings = lightings.to(self.device)
        nmap = nmap.to(self.device)

        single_lightings = lightings[:, 5, :, :, :] # [bs, 3, 256, 256]
        # repeat_nmaps = nmap.repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]
        
        # rgb
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_unet_f(rgb_latents, rgb_text_embs)
        B, C, H, W = rgb_unet_f.shape
        test_rgb_unet_f = torch.mean(rgb_unet_f.view(-1, 6, C, H, W), dim=1)
        test_rgb_unet_f = test_rgb_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        rgb_s, rgb_smap = self.compute_align_map(test_rgb_unet_f, self.patch_lib, rgb_unet_f.shape[-2:])
        
        # nromal map
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_controlnet_f(nmap_latents, single_lightings, nmap_text_embs)
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
        # repeat_nmaps = nmap.repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]
        
        # rgb
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_unet_f(rgb_latents, rgb_text_embs)
        B, C, H, W = rgb_unet_f.shape
        test_rgb_unet_f = torch.mean(rgb_unet_f.view(-1, 6, C, H, W), dim=1)
        test_rgb_unet_f = test_rgb_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        rgb_s, rgb_smap = self.compute_s_s_map(test_rgb_unet_f, self.patch_lib, rgb_unet_f.shape[-2:])
        
        # normal map
        nmap = nmap.to(self.device)
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_controlnet_f(nmap_latents, single_lightings, nmap_text_embs)
        B, C, H, W = nmap_unet_f.shape
        test_nmap_unet_f = nmap_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        nmap_s, nmap_smap = self.compute_s_s_map(test_nmap_unet_f, self.nmap_patch_lib, nmap_unet_f.shape[-2:])


        ### Combine RGB and Nmap score map ###
        # s = torch.maximum(rgb_s * self.weight + self.bias, nmap_s)
        s = rgb_s * nmap_s
        #s = (rgb_s * self.weight + self.bias) + nmap_s
        # s = (rgb_s * self.weight + self.bias + nmap_s)

        # smap = torch.maximum((rgb_smap * self.weight + self.bias), nmap_smap)
        smap = rgb_smap * self.weight + self.bias + nmap_smap
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
