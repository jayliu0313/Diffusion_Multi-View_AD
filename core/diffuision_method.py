import torch
import numpy as np
import torch.nn.functional as F
import math

from core.reconstruct_method import Reconstruct_Method
from utils.utils import t2np
# from patchify import patchify
from utils.visualize_util import display_one_img, display_image
from diffusers import DDIMScheduler
from core.models.unet_model import MyUNet2DConditionModel
from transformers import CLIPTextModel, AutoTokenizer
from torchvision.transforms import transforms
from kornia.filters import gaussian_blur2d

from core.models.controllora import  ControlLoRAModel
from tqdm import tqdm
from torch.optim.adam import Adam
from utils.ptp_utils import *
from sklearn import random_projection
import torch.nn.functional as nnf

class Diffusion_Method(Reconstruct_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.diffusion_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.diffusion_id, subfolder="text_encoder")
        self.noise_scheduler = DDIMScheduler.from_pretrained(args.diffusion_id, subfolder="scheduler")
        self.num_inference_timesteps = int(len(self.noise_scheduler.timesteps) / args.step_size)
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)
        self.timesteps_list = self.noise_scheduler.timesteps[self.noise_scheduler.timesteps <= args.noise_intensity]
        print("num_inference_timesteps")
        print("ddim loop steps:", len(self.timesteps_list))
        print("Noise Intensity = ", self.timesteps_list)
        self.memory_noise_intensity = args.memory_noise_intensity
        self.unet = MyUNet2DConditionModel.from_pretrained(
        args.diffusion_id,
        subfolder="unet",
        revision=args.revision)
        
        if args.load_unet_ckpt is not None:
            self.unet.load_state_dict(torch.load(args.load_unet_ckpt, map_location=self.device))
            print("Load Diffusion Model Checkpoint!!")

        self.unet.to(self.device)
        self.text_encoder.to(self.device)
        
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.unet.eval()
        self.text_encoder.eval()
          
        # Prepare text embedding
        self.uncond_embeddings = self.get_text_embedding("", 6) # [6, 77, 768]
            
    def forward_process_with_T(self, latents, T):
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        timesteps = torch.tensor([T], device=self.device).repeat(bsz)
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        return noise, timesteps, noisy_latents

    def get_text_embedding(self, text_prompt, bsz):
        tok = self.tokenizer(text_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embedding = self.text_encoder(tok.input_ids.to(self.device))[0].repeat((bsz, 1, 1))
        # print(text_embedding.shape)
        return text_embedding
    
    def pixel_distance(self, output, target):
        '''
        Pixel distance between image1 and image2
        '''
        output = self.image_transform(output)
        target = self.image_transform(target)
        distance_map = torch.mean(torch.abs(output - target), dim=1).unsqueeze(1)
        return distance_map
    
    def feature_distance(self, output, target):
        '''
        Feature distance between output and target
        '''
        target = self.image_transform(target)
        output = self.image_transform(output)
        inputs_features = self.feature_extractor(target)
        output_features = self.feature_extractor(output)
        print(inputs_features.shape)
    
        out_size = self.image_size
        anomaly_map = torch.zeros([inputs_features[0].shape[0] ,1 ,out_size, out_size]).to(self.device)
        for i in range(len(inputs_features)):
            if i == 0:
                continue
            # a_map = 1 - F.cosine_similarity(patchify(inputs_features[i]), patchify(output_features[i]))
            a_map = F.mse_loss(patchify(inputs_features[i]), patchify(output_features[i]))
            print(a_map)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
            anomaly_map += a_map
        return anomaly_map 
    
    def heat_map(self, output, target):
        '''
        Compute the anomaly map
        :param output: the output of the reconstruction
        :param target: the target image
        :param FE: the feature extractor
        :param sigma: the sigma of the gaussian kernel
        :param i_d: the pixel distance
        :param f_d: the feature distance
        '''
        sigma = 4
        kernel_size = 2 * int(4 * sigma + 0.5) +1
        anomaly_map = 0

        output = output.to(self.device)
        target = target.to(self.device)

        i_d = self.pixel_distance(output, target)
        f_d = self.feature_distance((output),  (target))
        f_d = torch.Tensor(f_d).to(self.device)
        # print('image_distance max : ',torch.max(i_d))
        # print('feature_distance max : ',torch.max(f_d))
        # visualalize_distance(output, target, i_d, f_d)
        anomaly_map += f_d + 6.0 *  torch.max(f_d)/ torch.max(i_d) * i_d  
        anomaly_map = gaussian_blur2d(
            anomaly_map , kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma)
            )
        anomaly_map = torch.sum(anomaly_map, dim=1).unsqueeze(1)
        return anomaly_map
        
class DDIM_Rec(Diffusion_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
    
    def reconstruct_latents(self, latents, text_embeddings):
        '''
        The reconstruction process
        :param y: the target image
        :param x: the input image
        :param seq: the sequence of denoising steps
        :param model: the UNet model
        :param x0_t: the prediction of x0 at time step t
        '''

        # Convert images to latent space
        #latents = self.get_vae_latents(lightings)
        bsz = latents.shape[0]

        # Add noise to the latents according to the noise magnitude at each timestep
        _, timesteps, noisy_latents = self.forward_process_with_T(latents, self.timesteps_list[0].item()) 

        
        # # Denoising Loop
        # self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        
        # for t in self.timesteps_list:
            
        #     timesteps = torch.tensor([t.item()], device=self.device).repeat(bsz)
        #     timesteps = timesteps.long()
        #     noise_pred = self.unet(
        #         noisy_latents,
        #         timesteps,
        #         encoder_hidden_states=encoder_hidden_states,
        #     )['sample']
        
        #     noisy_latents = self.noise_scheduler.step(noise_pred.to(self.device), t.item(), noisy_latents.to(self.device)).prev_sample
        
        return noisy_latents
    
    def predict(self, item, lightings, nmaps, text_prompt, gt, label):
        lightings = lightings.to(self.device).view(-1, 3, self.image_size, self.image_size) # [6, 3, 256, 256]
        #fc_lightings = self.get_FC_lightings(lightings)
        # nmaps = nmaps.to(self.device).repeat_interleave(6, dim=0) # [6, 3, 256, 256]
        latents = self.image2latents(lightings)
        rec_latents = self.reconstruct_latents(latents, text_prompt)
        rec_lightings = self.latents2image(rec_latents)
        if True:
            latents = latents.permute(0, 2, 3, 1)
            rec_latents = rec_latents.permute(0, 2, 3, 1)
            final_map = self.pdist(latents, rec_latents)
            if self.score_type == 0:
                final_map = torch.mean(final_map, dim=0)
                final_score = torch.max(final_map)
            else:
                final_map, idx = torch.max(final_map, dim=0)
                # print(s_map_size28.shape)
                final_score = final_map[idx]
            H, W = final_map.size()
            final_map = final_map.view(1, 1, H, W)
            final_map = torch.nn.functional.interpolate(final_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        else:
            loss = self.criteria(lightings, rec_lightings)
            self.cls_rec_loss += loss.item()
            final_map = torch.sum(torch.abs(lightings - rec_lightings), dim=1)

            if(self.score_type == 0):
                final_map, final_score, img = self.compute_max_smap(final_map, lightings)
            elif(self.score_type == 1):
                final_map, final_score, img = self.compute_mean_smap(final_map, lightings)
    
        
        # final_map = self.blur(final_map)
        img = rec_lightings[5]
        # self.cls_rec_loss += loss.item()
        self.image_labels.append(label)
        self.image_preds.append(t2np(final_score))
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(final_map))
        self.pixel_labels.extend(t2np(gt))
        display_image(lightings, rec_lightings, self.reconstruct_path, item)
        
class ControlNet_Rec(Diffusion_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        # Create ControlNet Model 
        self.controllora: ControlLoRAModel
        self.controllora = ControlLoRAModel.from_unet(self.unet, lora_linear_rank=args.controllora_linear_rank, lora_conv2d_rank=args.controllora_conv2d_rank)
        self.controllora.to(self.device)
        if args.load_controlnet_ckpt != None:
            self.controllora.load_state_dict(torch.load(args.load_controlnet_ckpt, map_location=self.device))
        self.controllora.tie_weights(self.unet)
        self.controllora.requires_grad_(False)
        self.controllora.eval()
        
    def reconstruction(self, lightings, nmaps, encoder_hidden_states):
        '''
        The reconstruction process
        :param y: the target image
        :param x: the input image
        :param seq: the sequence of denoising steps
        :param model: the UNet model
        :param x0_t: the prediction of x0 at time step t
        '''
        with torch.no_grad():
            # Convert images to latent space
            #latents = self.get_vae_latents(lightings)
            latents = self.image2latents(lightings)
            fc_latents = self.decomp_block.get_fc(latents)
            fu_latents = self.decomp_block.get_fu(latents)
            bsz = fc_latents.shape[0]

            # Add noise to the latents according to the noise magnitude at each timestep
            _, timesteps, noisy_latents = self.forward_process_with_T(fc_latents, self.timesteps_list[0].item()) 

            
            #encoder_hidden_states = lighting_text_embedding # [bs, 77, 768]
            condition_image = nmaps
            # Denoising Loop
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
            
            for t in self.timesteps_list:
                
                timesteps = torch.tensor([t.item()], device=self.device).repeat(bsz)
                timesteps = timesteps.long()
                down_block_res_samples, mid_block_res_sample = self.controllora(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=condition_image,
                    guess_mode=False,
                    return_dict=False,
                )

                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[sample for sample in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample
                ).sample
            
                noisy_latents = self.noise_scheduler.step(noise_pred.to(self.device), t.item(), noisy_latents.to(self.device)).prev_sample
            
            rec_latents = self.decomp_block.fuse_both(noisy_latents, fu_latents)
            rec_lightings = self.latents2image(rec_latents)
        return rec_lightings
        
    def predict(self, item, lightings, nmaps, gt, label):
        lightings = lightings.to(self.device).view(-1, 3, self.image_size, self.image_size) # [6, 3, 256, 256]
        #fc_lightings = self.get_FC_lightings(lightings)
        nmaps = nmaps.to(self.device).repeat_interleave(6, dim=0) # [6, 3, 256, 256]
        rec_lightings = self.reconstruction(lightings, nmaps, self.encoder_hidden_states)
        # final_map = self.heat_map(rec_lightings, lightings)
        # final_map, final_score, img = self.compute_max_smap(final_map, rec_lightings)
        # final_score = torch.max(final_map)
        
        final_map, final_score = self.compute_feature_dist(lightings, rec_lightings)
        img = rec_lightings[5]
        # self.cls_rec_loss += loss.item()
        self.image_labels.append(label)
        self.image_preds.append(t2np(final_score))
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(final_map))
        self.pixel_labels.extend(t2np(gt))
        if item % 2 == 0:
            display_image(t2np(lightings), t2np(rec_lightings), self.reconstruct_path, item)

class DDIMInv_Rec(Diffusion_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.guidance_scale = args.guidance_scale
        self.num_inner_steps = args.num_opt_steps
        self.opt_max_steps = args.opt_max_steps
        print("num optimize steps:", self.num_inner_steps)
        print("guidance scale:", self.guidance_scale)
           
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.noise_scheduler.config.num_train_timesteps // self.noise_scheduler.num_inference_steps
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.noise_scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.noise_scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.noise_scheduler.config.num_train_timesteps // self.noise_scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.noise_scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.noise_scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.guidance_scale
        noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents
        
    @torch.no_grad()
    def ddim_loop(self, latent, cond_embeddings):
        # uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        for t in reversed(self.timesteps_list):
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent
    
    @property
    def scheduler(self):
        return self.noise_scheduler

    def null_optimization(self, latents, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        num_ddim_loop = len(self.timesteps_list)
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        for i, t in enumerate(self.timesteps_list):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / (num_ddim_loop * 2)))
            latent_prev = latents[len(latents) - i - 2]
            # mean_uncond = torch.mean(uncond_embeddings, dim=0)
            # mean_uncond = mean_uncond.repeat((6, 1, 1))
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(self.num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)

                # dev_loss = nnf.mse_loss(mean_uncond, uncond_embeddings)
                # print("dev loss:", dev_loss)
                latent_loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                # print("latent_loss:", latent_loss)
                # print(mean_uncond)
                # print(dev_loss)
                loss = latent_loss #+ dev_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                if loss_item < epsilon + i * 2e-5:
                    break

            # print("null_optimize_uncondition embedd", uncond_embeddings.shape)
            uncond_embeddings_list.append(uncond_embeddings.detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        # bar.close()
        return uncond_embeddings_list

    def invert(self, latents, cond_embeddings, early_stop_epsilon=1e-5, verbose=False):
        # register_attention_control(self.model, None)
        self.context = torch.cat([self.uncond_embeddings, cond_embeddings])
        if verbose:
            print("DDIM inversion...")
        ddim_latents = self.ddim_loop(latents, cond_embeddings)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, early_stop_epsilon)
        return ddim_latents[-1], uncond_embeddings
    
    def reconstruction(
        self,
        noisy_latents,
        uncond_embeddings,
        cond_embeddings
    ):
        # latent, latents = init_latent(latent, model, height, width, generator, batch_size)
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        # print("input", noisy_latents.shape)
        for i, t in enumerate(self.timesteps_list):
            timesteps = torch.tensor(t.item(), device=self.device)
            context = torch.cat([uncond_embeddings[i], cond_embeddings])
            latents_input = torch.cat([noisy_latents] * 2)
            noise_pred = self.unet(
                latents_input,
                timesteps,
                encoder_hidden_states=context,
            )['sample']
            # print(noise_pred.shape)
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_prediction_text - noise_pred_uncond)
            noisy_latents = self.noise_scheduler.step(noise_pred, t.item(), noisy_latents)["prev_sample"]
        return noisy_latents
    
    def predict(self, item, lightings, nmaps, text_prompt, gt, label):
        lightings = lightings.to(self.device).view(-1, 3, self.image_size, self.image_size) # [6, 3, 256, 256]
        cond_embeddings = self.get_text_embedding(text_prompt, 6)
    
        with torch.no_grad():
            latents = self.image2latents(lightings)
        
        ddim_latent, uncond_embeddings = self.invert(latents, cond_embeddings)
        # print(ddim_latent.shape)
        with torch.no_grad():
            rec_latents = self.reconstruction(ddim_latent, uncond_embeddings, cond_embeddings)
            sampled_images = self.latents2image(rec_latents)
            
        latents = latents.permute(0, 2, 3, 1)
        rec_latents = rec_latents.permute(0, 2, 3, 1)
        final_map = self.pdist(latents, rec_latents)
        
        if self.score_type == 0:
            final_map = torch.mean(final_map, dim=0)
            final_score = torch.max(final_map)
        else:
            final_map, idx = torch.max(final_map, dim=0)
            # print(s_map_size28.shape)
            final_score = final_map[idx]

        H, W = final_map.size()
        final_map = final_map.view(1, 1, H, W).cpu()
        
        final_map = torch.nn.functional.interpolate(final_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        final_map = self.blur(final_map)
        img = lightings[5]
        # self.cls_rec_loss += loss.item()
        self.image_labels.append(label)
        self.image_preds.append(t2np(final_score))
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(final_map))
        self.pixel_labels.extend(t2np(gt))

        display_image(lightings, sampled_images, self.reconstruct_path, item)

class DDIM_Memory(Diffusion_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        
        self.f_coreset = 1
        self.coreset_eps = 0.9
        self.n_reweight = 3
        
    def compute_s_s_map(self, patch, feature_map_dims):
        # patch = (patch - self.mean)/self.std
        # self.patch_lib = self.rgb_layernorm(self.patch_lib)
        # print(patch.shape)
        # print(self.patch_lib.shape)
        # patch = patch.to('cpu')
        self.patch_lib = self.patch_lib.to(self.device)
        dist = torch.cdist(patch, self.patch_lib)
 
        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = w * s_star
        
        # segmentation map
        # print(min_val)
        
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')
        s_map = self.blur(s_map.to('cpu'))
        return s.to('cpu'), s_map

    def get_memory_nnfeature(self, patch, feature_map_dims):
        dist = torch.cdist(patch, self.patch_lib.to(self.device))
        min_idx = torch.argmin(dist, dim=1)
        nnfeature = self.patch_lib[min_idx].view(1, 4, *feature_map_dims)
        nnfeature = nnfeature.repeat(6, 1, 1, 1)
        # print(nnfeature.shape)
        return nnfeature.to(self.device)
    
    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0)
        # self.mean = torch.mean(self.patch_lib)
        # self.std = torch.std(self.patch_lib)
        # self.patch_lib = (self.patch_lib - self.mean)/self.std

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
   
    def add_sample_to_mem_bank(self, lightings, text_prompt):
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        latents = self.image2latents(lightings)
        text_embeddings = self.get_text_embedding(text_prompt, 6)
        _, timesteps, noisy_latents = self.forward_process_with_T(latents, self.memory_noise_intensity)
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
        with torch.no_grad():
            latents = self.image2latents(lightings)
            cond_embeddings = self.get_text_embedding(text_prompt, 6)

        with torch.no_grad():
            _, timesteps, noisy_latents = self.forward_process_with_T(latents, self.memory_noise_intensity)
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

class DDIMInv_Memory(DDIMInv_Rec):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        
        self.f_coreset = 1
        self.coreset_eps = 0.9
        self.n_reweight = 3
        
    def compute_s_s_map(self, patch, feature_map_dims):
        # patch = (patch - self.mean)/self.std
        # self.patch_lib = self.rgb_layernorm(self.patch_lib)
        # print(patch.shape)
        # print(self.patch_lib.shape)
        # patch = patch.to('cpu')
        self.patch_lib = self.patch_lib.to(self.device)
        dist = torch.cdist(patch, self.patch_lib)
 
        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        # s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        # m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        # m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        # w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        # _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        # m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        # D = torch.sqrt(torch.tensor(patch.shape[1]))
        # w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        # s = w * s_star
        
        # segmentation map
        # print(min_val)
        
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
        # self.mean = torch.mean(self.patch_lib)
        # self.std = torch.std(self.patch_lib)
        # self.patch_lib = (self.patch_lib - self.mean)/self.std

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
    
    def get_unet_latent(self, noisy_latents, timesteps, cond_embeddings, layer=3):
        # uncond_embeddings, cond_embeddings = self.context.chunk(2)
        model_output = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=cond_embeddings,
        )
        return model_output['up_ft'][layer]
    
    def add_sample_to_mem_bank(self, lightings, text_prompt):
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        latents = self.image2latents(lightings)
        text_embeddings = self.get_text_embedding(text_prompt, 6)
        
        ddim_latents = self.ddim_loop(latents, text_embeddings)
        # print("timestep:",self.timesteps_list[0])-1
        unet_f = self.get_unet_latent(ddim_latents[-1], self.timesteps_list[0], text_embeddings)
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

            ddim_latents = self.ddim_loop(latents, text_embeddings)
        
            unet_f = self.get_unet_latent(ddim_latents[-1], self.timesteps_list[0], text_embeddings)
        
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