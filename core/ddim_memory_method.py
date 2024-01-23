import torch

from core.base import DDIM_Method
from utils.utils import t2np
from utils.visualize_util import display_one_img, display_image
from core.models.controllora import  ControlLoRAModel
from torch.optim.adam import Adam
import torch.nn.functional as nnf

from tqdm import tqdm
from utils.ptp_utils import *

class Memory_Method(DDIM_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.f_coreset = 1
        self.coreset_eps = 0.9
        self.n_reweight = 3
        self.memory_intensity = args.memory_intensity
        
    def compute_s_s_map(self, patch, feature_map_dims):
        self.patch_lib = self.patch_lib.to(self.device)
        dist = torch.cdist(patch, self.patch_lib)
        # print(self.patch_lib.shape)
        min_val, min_idx = torch.min(dist, dim=1)
        # print(min_idx.shape)
        # print(min_idx)
        # a = self.patch_lib[min_idx]
        
        s_star = torch.max(min_val)
        
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')
        s_map = self.blur(s_map.to('cpu'))
        return s_star.to('cpu'), s_map
    
    def get_memory_nnfeature(self, patch, feature_map_dims):
        dist = torch.cdist(patch, self.patch_lib.to(self.device))
        min_idx = torch.argmin(dist, dim=1)
        nnfeature = self.patch_lib[min_idx].view(1, 4, *feature_map_dims)
        nnfeature = nnfeature.repeat(6, 1, 1, 1)
        # print(nnfeature.shape)
        return nnfeature.to(self.device)
    
    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0)
        
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

        s, smap = self.compute_s_s_map(test_unet_f, unet_f.shape[-2:])
        img = lightings[5, :, :, :]
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(smap))
        self.pixel_labels.extend(t2np(gt))

class DDIMInv_Memory(Memory_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)

    @torch.no_grad()
    def ddim_loop(self, latent, cond_embeddings, is_certain_intensity):
        # uncond_embeddings, cond_embeddings = self.context.chunk(2)
        latent = latent.clone().detach()
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        for t in reversed(self.timesteps_list):
            timesteps = torch.tensor(t.item(), device=self.device)
            pred = self.unet(latent, t, encoder_hidden_states=cond_embeddings)
            latent = self.next_step(pred['sample'], t, latent)
            if is_certain_intensity == True and self.memory_intensity == t:
                return latent, t
        return latent, self.timesteps_list[0]
    
    def diffusion_loop(self, noisy_latents, text_emb):
        # latent, latents = init_latent(latent, model, height, width, generator, batch_size)
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        # print("input", noisy_latents.shape)
        for i, t in enumerate(self.timesteps_list):
            timesteps = torch.tensor(t.item(), device=self.device)
            pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=text_emb)
            noisy_latents = self.noise_scheduler.step(pred['sample'], t.item(), noisy_latents)["prev_sample"]
            if t == self.memory_intensity:
                return pred['up_ft'][3]
                        
    def get_unet_latent(self, noisy_latents, timesteps, cond_embeddings, layer=3):
        # uncond_embeddings, cond_embeddings = self.context.chunk(2)
        model_output = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=cond_embeddings,
        )
        return model_output['up_ft'][layer]
    
    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        latents = self.image2latents(lightings)
        bsz = latents.shape[0]
        text_embeddings = self.get_text_embedding(text_prompt, bsz)
        
        ddim_latents, timestep = self.ddim_loop(latents, text_embeddings, True)
        # print("timestep:",self.timesteps_list[0])-1
        unet_f = self.get_unet_latent(ddim_latents, timestep, text_embeddings)
        B, C, H, W = unet_f.shape
        
        train_unet_f = torch.mean(unet_f.view(-1, 6, C, H, W), dim=1)
        train_unet_f = train_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        self.patch_lib.append(train_unet_f.cpu())
    
    def predict(self, i, lightings, nmap, text_prompt, gt, label):
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        with torch.no_grad():
            latents = self.image2latents(lightings)
            text_embeddings = self.get_text_embedding(text_prompt, 6)

            ddim_latents, timestep = self.ddim_loop(latents, text_embeddings, False)
        
            unet_f = self.diffusion_loop(ddim_latents, text_embeddings)
        
        B, C, H, W = unet_f.shape
        
        test_unet_f = torch.mean(unet_f.view(-1, 6, C, H, W), dim=1)
        test_unet_f = test_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T

        s, smap = self.compute_s_s_map(test_unet_f, unet_f.shape[-2:])
        img = lightings[5, :, :, :]
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(smap))
        self.pixel_labels.extend(t2np(gt))
        
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
        
        self.f_coreset = 1.0
        self.coreset_eps = 0.9
        self.n_reweight = 3
        self.optimize_loop = 5
                
    def controlnet(self, noisy_latents, nmap, timestep, text_emb):
        down_block_res_samples, mid_block_res_sample = self.controllora(
            noisy_latents, timestep,
            encoder_hidden_states=text_emb,
            controlnet_cond=nmap,
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
    def ddim_loop(self, latents, nmap, text_emb, is_certain_intensity):
        latents = latents.clone().detach()
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        for t in reversed(self.timesteps_list):
            timesteps = torch.tensor(t.item(), device=self.device)
            pred = self.controlnet(latents, nmap, timesteps, text_emb)
            latents = self.next_step(pred['sample'], t, latents)
            if is_certain_intensity == True and self.memory_intensity == t:
                return latents, t
        return latents, self.timesteps_list[0]
     
    def diffusion_loop(self, noisy_latents, nmap, text_emb):
        # latent, latents = init_latent(latent, model, height, width, generator, batch_size)
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        # print("input", noisy_latents.shape)
        for i, t in enumerate(self.timesteps_list):
            timesteps = torch.tensor(t.item(), device=self.device)
            pred = self.controlnet(noisy_latents, nmap, timesteps, text_emb)
            noisy_latents = self.noise_scheduler.step(pred['sample'], t.item(), noisy_latents)["prev_sample"]
            if t == self.memory_intensity:
                return pred['up_ft'][3]
            
    def condition_optimization(self, unet_f, latent, nmap, text_embedding, epsilon=1e-5):
        self.patch_lib = self.patch_lib.to(self.device)
        dist = torch.cdist(unet_f, self.patch_lib)
        # print(self.patch_lib.shape)
        min_val, min_idx = torch.min(dist, dim=1)
        memory_feature = self.patch_lib[min_idx]
        
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)

        opt_text_embedding = text_embedding.clone().detach()
        opt_text_embedding.requires_grad = True
        optimizer = Adam([opt_text_embedding], lr=1e-4)

        # latent_prev = latents[len(latents) - i - 2]
        for j in range(self.optimize_loop):
            opt_unet_f = self.controlnet(latent, nmap, self.timesteps_list[0], opt_text_embedding)['up_ft'][3]
            B, C, H, W = opt_unet_f.shape
            opt_unet_f = torch.mean(opt_unet_f.view(-1, 6, C, H, W), dim=1)
            opt_unet_f = opt_unet_f.reshape(C, -1).T
            latent_loss = nnf.mse_loss(opt_unet_f, memory_feature)
            # print("latent_loss:", latent_loss)

            loss = latent_loss #+ dev_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item = loss.item()
            
        s_map = self.pdist(opt_unet_f, memory_feature)
        min_val, _ = torch.min(dist, dim=1)
        s_star = torch.max(min_val)
        feat_num = s_map.shape[0]
        size = int(np.sqrt(feat_num))
        s_map = s_map.view(1, 1, size, size)
        s_map_size_img = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map_size_img = self.blur(s_map_size_img.to('cpu'))
        return s_star.to('cpu'), s_map_size_img
        
    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):
        lightings = lightings.to(self.device).view(-1, 3, self.image_size, self.image_size) # [B * 6, 3, 256, 256]
        nmaps = nmap.to(self.device).repeat_interleave(6, dim=0) # [B * 6, 3, 256, 256]
        
        latents = self.image2latents(lightings)
        bs = lightings.shape[0]
        text_emb = self.get_text_embedding(text_prompt, bs)
        
        ddim_latents, timestep = self.ddim_loop(latents, nmaps, text_emb, False)
        # unet_f = self.controlnet(ddim_latents, nmaps, timestep, text_emb)['up_ft'][3]
        unet_f = self.diffusion_loop(ddim_latents, nmaps, text_emb)
        
        B, C, H, W = unet_f.shape
        train_unet_f = torch.mean(unet_f.view(-1, 6, C, H, W), dim=1).squeeze(dim=1)
        train_unet_f = train_unet_f.permute((1, 0, 2, 3)).reshape(C, -1).T
        self.patch_lib.append(train_unet_f.cpu())
        
    def predict(self, i, lightings, nmap, text_prompt, gt, label):
        lightings = lightings.to(self.device).view(-1, 3, self.image_size, self.image_size) # [6, 3, 256, 256]
        nmaps = nmap.to(self.device).repeat_interleave(6, dim=0) # [6, 3, 256, 256]
        
        latents = self.image2latents(lightings)

        text_emb = self.get_text_embedding(text_prompt, 6)
        ddim_latents, _ = self.ddim_loop(latents, nmaps, text_emb, False)
        unet_f = self.diffusion_loop(ddim_latents, nmaps, text_emb)
        # unet_f = self.controlnet(ddim_latents, nmaps, timestep, text_emb)['up_ft'][3]
        
        B, C, H, W = unet_f.shape
        test_unet_f = torch.mean(unet_f.view(-1, 6, C, H, W), dim=1)
        test_unet_f = test_unet_f.reshape(C, -1).T
        
        # s, s_map = self.condition_optimization(test_unet_f, noisy_latents, nmaps, encoder_hidden_states)
        s, s_map = self.compute_s_s_map(test_unet_f, unet_f.shape[-2:])
        self.image_list.append(t2np(lightings[5]))
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.pixel_labels.extend(t2np(gt))
        self.pixel_preds.append(t2np(s_map))