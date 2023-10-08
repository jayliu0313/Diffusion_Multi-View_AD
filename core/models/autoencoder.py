import numpy as np
import torch
import torch.nn as nn
from core.models.backnone import RGB_Extractor, ResNet_Dec
from core.models.network_util import *



class Autoencoder(nn.Module):
    def __init__(self, device, latent_dim=256):
        super(Autoencoder, self).__init__()
        self.feature_extractor = RGB_Extractor(device)
        self.feature_extractor.freeze_parameters(layers=[], freeze_bn=True)
        feature_dims = [64, 128, 256]
        in_dim = sum(feature_dims)
        self.scale_factors = [
            in_stride / 128 for in_stride in feature_dims
        ]  # for resize
        self.upsample_list = [
            nn.UpsamplingBilinear2d(scale_factor=scale_factor)
            for scale_factor in self.scale_factors
        ]
        self.feature_mapping = nn.Conv2d(in_dim, latent_dim, 3, 1, 1)
        self.decomp = Decom_Block(latent_dim)
        # self.feature_mappings = [
        #     nn.Conv2d()
        # ]
        self.decoder = ResNet_Dec(256, 16)
        
    def forward(self, x):
        rgb_feature_maps = self.feature_extractor(x)
        
        feature_list = []
        
        # resize & concatenate
        for i in range(len(rgb_feature_maps)):
            upsample = self.upsample_list[i]
            feature_resize = upsample(rgb_feature_maps[i])
            print(feature_resize.shape)
            feature_list.append(feature_resize)
            
        cat_feature = torch.cat(feature_list, dim=1)
        align_feature = self.feature_mapping(cat_feature)
        fuse_feature = self.decomp(align_feature)
        out = self.decoder(fuse_feature)
        return out