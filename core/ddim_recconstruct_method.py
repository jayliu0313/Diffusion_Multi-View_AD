import torch
from core.base import DDIM_Method
from utils.utils import t2np
from utils.visualize_util import display_one_img, display_image
from core.models.controllora import  ControlLoRAModel
from torch.optim.adam import Adam
import torch.nn.functional as nnf
import os

class Reconstruct_Method(DDIM_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
                
    def compute_feature_dist(self, target, reconstruct_img):
        feat1 = self.get_rgb_feature(target)
        feat2 = self.get_rgb_feature(reconstruct_img)
        
        feat1 = feat1.permute(0, 2, 3, 1)
        feat2 = feat2.permute(0, 2, 3, 1)
        s_map_size28 = self.pdist(feat1, feat2)
        if self.score_type == 0:
            s_map_size28 = torch.mean(s_map_size28, dim=0)
            s = torch.max(s_map_size28)
        else:
            s_map_size28, idx = torch.max(s_map_size28, dim=0)
            # print(s_map_size28.shape)
            s = s_map_size28[idx]
        # print(s_map_size28.shape)
        s_map_size28 = s_map_size28.view(1, 1, 28, 28)
        s_map_size_img = torch.nn.functional.interpolate(s_map_size28, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        return s_map_size_img.to('cpu'), s.to('cpu')

    def compute_max_smap(self, score_maps, lightings):
        # final_map = torch.zeros((1, self.image_size, self.image_size))
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
    
class DDIM_Rec(Reconstruct_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
    
    def compute_max_smap(self, score_maps, lightings):
        # final_map = torch.zeros((1, self.image_size, self.image_size))
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
    
    def predict(self, i, lightings, nmaps, text_prompt, gt, label):
        lightings = lightings.to(self.device).view(-1, 3, self.image_size, self.image_size) # [6, 3, 256, 256]
        # nmaps = nmaps.to(self.device).repeat_interleave(6, dim=0) # [6, 3, 256, 256]
        latents = self.image2latents(lightings)

        # Add noise to the latents according to the noise magnitude at each timestep
        # print(self.timesteps_list[0])
        _, _, rec_latents = self.forward_process_with_T(latents, self.timesteps_list[0].item()) 
        
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
        display_image(lightings, rec_lightings, self.reconstruct_path, i)

class ControlNet_Rec(Reconstruct_Method):
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
        
    def reconstruction(self, latents, nmaps, encoder_hidden_states):
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
        
            bsz = latents.shape[0]
        
            # Add noise to the latents according to the noise magnitude at each timestep
            _, timesteps, noisy_latents = self.forward_process_with_T(latents, self.timesteps_list[0].item()) 

            
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
                )["sample"]
            
                noisy_latents = self.noise_scheduler.step(noise_pred.to(self.device), t.item(), noisy_latents.to(self.device)).prev_sample
        return noisy_latents
        
    def predict(self, i, lightings, nmaps, text_prompt, gt, label):
        lightings = lightings.to(self.device).view(-1, 3, self.image_size, self.image_size) # [6, 3, 256, 256]
        
        nmaps = nmaps.to(self.device).repeat_interleave(6, dim=0) # [6, 3, 256, 256]
        text_embeddings = self.get_text_embedding(text_prompt, 6)
        latents = self.image2latents(lightings)
        rec_latents = self.reconstruction(latents, nmaps, text_embeddings)
        rec_lightings = self.latents2image(rec_latents)
        # final_map = self.heat_map(rec_lightings, lightings)
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
        # final_score = torch.max(final_map)
        
        # final_map, final_score = self.compute_feature_dist(lightings, rec_lightings)
        img = rec_lightings[5]
        # self.cls_rec_loss += loss.item()
        self.image_labels.append(label)
        self.image_preds.append(t2np(final_score))
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(final_map))
        self.pixel_labels.extend(t2np(gt))
        if i % 2 == 0:
            display_image(lightings, rec_lightings, self.reconstruct_path, i)

class NULLInv_Rec(Reconstruct_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.guidance_scale = args.guidance_scale
        self.num_inner_steps = args.num_opt_steps
        self.opt_max_steps = args.opt_max_steps
        print("num optimize steps:", self.num_inner_steps)
        print("guidance scale:", self.guidance_scale)
           
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
    
    def predict(self, i, lightings, nmaps, text_prompt, gt, label):
        lightings = lightings.to(self.device).view(-1, 3, self.image_size, self.image_size) # [6, 3, 256, 256]
        cond_embeddings = self.get_text_embedding(text_prompt, 6)
    
        
        latents = self.image2latents(lightings)
        ddim_latent, uncond_embeddings = self.invert(latents, cond_embeddings)
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

        display_image(lightings, sampled_images, self.reconstruct_path, i)