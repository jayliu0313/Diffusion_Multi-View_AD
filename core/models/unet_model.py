import torch
import torch.nn as nn
from core.models.network_util import *
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers import UNet2DConditionModel

class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Tuple:

        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            # SDXL - style
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb, hint = self.add_embedding(image_embs, hint)
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = mid_block_additional_residual is None and down_block_additional_residuals is not None

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_block_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_block_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, scale=lora_scale)

                if is_adapter and len(down_block_additional_residuals) > 0:
                    sample += down_block_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_block_additional_residuals) > 0
                and sample.shape == down_block_additional_residuals[0].shape
            ):
                sample += down_block_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    scale=lora_scale,
                )
            
            up_ft[i] = sample

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        output={}
        output['sample'] = sample
        output['up_ft'] = up_ft
        return output

# def double_conv(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(out_channels, out_channels, 3, padding=1),
#         nn.ReLU(inplace=True)
#     )   

# def gauss_noise_tensor(img):
#     assert isinstance(img, torch.Tensor)
#     dtype = img.dtype
#     if not img.is_floating_point():
#         img = img.to(torch.float32)
    
#     sigma = 0.2
    
#     out = img + sigma * torch.randn_like(img)
    
#     if out.dtype != dtype:
#         out = out.to(dtype)
        
#     return out

class UNet_Decom(nn.Module):

    def __init__(self, device, ch=32, mid_ch=16):
        super(UNet_Decom, self).__init__()
        self.device = device
        img_ch = 3 

        self.dconv_down1 = double_conv(img_ch, ch)
        self.decom_block1 = Decom_Block(ch, mid_ch)
        self.dconv_down2 = double_conv(ch, ch*2)
        self.decom_block2 = Decom_Block(ch*2, mid_ch*2)
        self.dconv_down3 = double_conv(ch*2, ch*4)
        self.decom_block3 = Decom_Block(ch*4, mid_ch*4)
        self.dconv_down4 = double_conv(ch*4, ch*8)   
        self.decom_block4 = Decom_Block(ch*8, mid_ch*8)     

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        # self.bottleneck1 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(28*28*256, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, self.feature_dim),
        # )

        # self.bottleneck2 = nn.Sequential(
        #     nn.Linear(self.feature_dim, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 28*28*256),
        #     nn.Unflatten(1, (256, 28, 28)),
        # )

        self.dconv_up3 = double_conv(ch*8 + ch*4, ch*4)
        self.dconv_up2 = double_conv(ch*4 + ch*2, ch*2)
        self.dconv_up1 = double_conv(ch*2 + ch, ch)
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 3, 1, 1),
            nn.Sigmoid(),                            
        )
 
    def forward(self, x):
        # _in = lightings.to(self.device)
        
        x = self.dconv_down1(x)
        conv1 = self.decom_block1(x)
        x = self.maxpool(conv1)

        x = self.dconv_down2(x)
        conv2 = self.decom_block2(x)
        x = self.maxpool(conv2)
        
        x = self.dconv_down3(x)
        conv3 = self.decom_block3(x)
        x = self.maxpool(conv3)   
    
        x = self.dconv_down4(x)
        x = self.decom_block4(x)
        
        # embedded = self.bottleneck1(x)
        
        # x = self.bottleneck2(embedded)

        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
    
        return out
    
    def rand_forward(self, x):
        x = self.dconv_down1(x)
        conv1 = self.decom_block1.rand_forward(x)
        x = self.maxpool(conv1)

        x = self.dconv_down2(x)
        conv2 = self.decom_block2.rand_forward(x)
        x = self.maxpool(conv2)
        
        x = self.dconv_down3(x)
        conv3 = self.decom_block3.rand_forward(x)
        x = self.maxpool(conv3)   
    
        x = self.dconv_down4(x)
        x = self.decom_block4.rand_forward(x)
        
        # embedded = self.bottleneck1(x)
        
        # x = self.bottleneck2(embedded)

        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
    
        return out
    
    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False


class ResUNet_Decom_AE(nn.Module):
    def __init__(self, device, ch=32, mid_ch=16):
        super(ResUNet_Decom_AE, self).__init__()
        self.device = device
        img_ch = 3
        self.img_size = 224
        
        self.dconv_down1 = BasicBlock(img_ch, ch, 2, short_cut=conv1x1(3, ch, 2))
        self.decom_block1 = Decom_Block(ch, mid_ch)
        
        self.dconv_down2 = BasicBlock(ch, ch*2, 2, short_cut=conv1x1(ch, ch*2, 2))
        self.decom_block2 = Decom_Block(ch*2, mid_ch*2)
        
        self.dconv_down3 = BasicBlock(ch*2, ch*4, 2, short_cut=conv1x1(ch*2, ch*4, 2))
        self.decom_block3 = Decom_Block(ch*4, mid_ch*4)
        
        self.dconv_down4 = BasicBlock(ch*4, ch*8, 2, short_cut=conv1x1(ch*4, ch*8, 2))     
        self.decom_block4 = Decom_Block(ch*8, mid_ch*8)
        
        upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)        
      
        self.dconv_up3 = BasicBlock(ch*8, ch*4, upsample=upsample, 
                                    short_cut=nn.Sequential(conv1x1(ch*8, ch*4, stride=1), upsample))
        self.dconv_up2 = BasicBlock(ch*8, ch*2, upsample=upsample,
                                    short_cut=nn.Sequential(conv1x1(ch*8, ch*2, stride=1), upsample))
        self.dconv_up1 = BasicBlock(ch*4, ch, upsample=upsample,
                                    short_cut=nn.Sequential(conv1x1(ch*4, ch, stride=1), upsample))
        self.conv_last = BasicBlock(ch*2, img_ch, upsample=upsample, 
                                    short_cut=nn.Sequential(conv1x1(ch*2, img_ch, stride=1), upsample))

    def forward(self, x):
        # add_random_masked(lighting)
        # if self.training:
        #     x = gauss_noise_tensor(x, 1.0)

        x = self.dconv_down1(x)
        conv1 = self.decom_block1(x)
        x = self.dconv_down2(conv1)
        conv2 = self.decom_block2(x)
        x = self.dconv_down3(conv2)
        conv3 = self.decom_block3(x)
        x = self.dconv_down4(conv3)
        x = self.decom_block4(x)

        x = self.dconv_up3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up1(x)
        x = torch.cat([x, conv1], dim=1)
        out = self.conv_last(x)
        return out
    
    def rand_forward(self, x):
        # add_random_masked(lighting)
        # if self.training:
        #     x = gauss_noise_tensor(x, 1.0)
       
        x = self.dconv_down1(x)
        conv1 = self.decom_block1.rand_forward(x)
        x = self.dconv_down2(conv1)
        conv2 = self.decom_block2.rand_forward(x)
        x = self.dconv_down3(conv2)
        conv3 = self.decom_block3.rand_forward(x)
        x = self.dconv_down4(conv3)
        x = self.decom_block4.rand_forward(x)
        
        x = self.dconv_up3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up1(x)
        x = torch.cat([x, conv1], dim=1)
        out = self.conv_last(x)
        return out

    def decode(self, x):
        x = self.dconv_up3(x)
        x = self.dconv_up2(x)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out

    def mean_rec(self, x):
        if self.training:
            x = gauss_noise_tensor(x, 1.0)

        x = self.encode(x)

        # if self.training:
        #     x = add_jitter(x, 30, 0.5)
        fc = self.common_MLP(x)
        fu = self.unique_MLP(x)  
        mean_fc = torch.mean(fc, dim = 0)
        mean_fc = mean_fc.repeat(6, 1, 1, 1)
        x = self.fuse_both(mean_fc + fu)
        out = self.decode(x)
        return out
    
    def get_meanfc(self, x):
        x = self.encode(x)
        fc = self.common_MLP(x)
        mean_fc = torch.mean(fc, dim = 0)
        return mean_fc
    
    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False
