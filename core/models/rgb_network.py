import torch.nn as nn
import torch
import random
from core.models.network_util import *
from utils.utils import KNNGaussianBlur

class Convolution_AE(nn.Module):

    def __init__(self, device, channel = 32):
        super(Convolution_AE, self).__init__()
        self.device = device
        self.image_chennels = 3
        self.img_size = 224
        self.dconv_down1 = double_conv(3, channel)
        self.dconv_down2 = double_conv(channel, channel * 2)
        self.dconv_down3 = double_conv(channel * 2, channel * 4)
        self.dconv_down4 = double_conv(channel * 4, channel * 8)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

        self.common_MLP = nn.Sequential(
            nn.Conv2d(channel * 8, channel * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(channel * 8, channel * 8, 1, 1),
        )

        self.unique_MLP = nn.Sequential(
            nn.Conv2d(channel * 8, channel * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(channel * 8, channel * 8, 1, 1),
            
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(channel * 8, channel * 8, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        
        self.dconv_up3 = double_conv(channel * 8, channel * 4)
        self.dconv_up2 = double_conv(channel * 4, channel * 2)
        self.dconv_up1 = double_conv(channel * 2, channel)
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, self.image_chennels, 1),
            nn.Sigmoid(),                            
        )
       
    def forward(self, lighting):
        fc, fu = self.encode(lighting)
        out = self.decode(fc, fu)
        return fc, out

    def encode(self, lighting):
        
        conv1 = self.dconv_down1(lighting.to(self.device))
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
    
        x = self.dconv_down4(x)
        
        fc = self.common_MLP(x)
        fu = self.unique_MLP(x)

        return fc, fu

    def decode(self, fc, fu):
        cat_feature = fc + fu
            
        x = self.fusion(cat_feature)

        x = self.upsample(x)        
        
        x = self.dconv_up3(x)
        x = self.upsample(x)              

        x = self.dconv_up2(x)
        x = self.upsample(x)
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        return out
     
    def get_fc(self, lighting):
        conv1 = self.dconv_down1(lighting.to(self.device))
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
    
        x = self.dconv_down4(x)
        
        fc = self.common_MLP(x)
        
        return fc
    
    def get_fu(self, lighting):
        conv1 = self.dconv_down1(lighting.to(self.device))
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
    
        x = self.dconv_down4(x)
        
        fu = self.unique_MLP(x)

        return fu

    def get_mean_fc(self, six_fc):
        fc = six_fc.reshape(-1, 6, 256, 28, 28)
        mean_fc = torch.mean(fc, dim = 1)
        mean_fc = mean_fc.reshape(-1, 256, 28, 28)
        return mean_fc
    
    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False

# V1
class Masked_ConvAE(nn.Module):
    def __init__(self, device, channel=32):
        super(Masked_ConvAE, self).__init__()
        self.device = device
        self.image_chennels = 3
        self.img_size = 224
        self.dconv_down1 = masked_double_conv(3, channel)
        self.dconv_down2 = masked_double_conv(channel, channel * 2)
        self.dconv_down3 = masked_double_conv(channel * 2, channel * 4)
        self.dconv_down4 = masked_double_conv(channel * 4, channel * 8)        
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

        self.common_MLP = nn.Sequential(
            MaskedConv2d_3x3(channel * 8, channel * 8, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 8, channel * 8, 1, 1),
        )
        
        self.fusion_fc = nn.Sequential(
            nn.Conv2d(channel * 8 * 6, channel * 8, 1, 1),
        )
        # self.fuse_fc = nn.Sequential(
        #     nn.Conv2d(channel * 8 * 6, channel * 8, 1, 1),
        # )

        self.unique_MLP = nn.Sequential(
            MaskedConv2d_3x3(channel * 8, channel * 8, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 8, channel * 8, 1, 1),
        )

        self.cross_atten = CrossAttention()

        # self.fuse_both = iAFF(channel * 8, 4)
        self.fusion = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 8, channel * 8, 1, 1),
            nn.ReLU(inplace=True),
        )
        
        self.dconv_up3 = masked_double_conv(channel * 8, channel * 4)
        self.dconv_up2 = masked_double_conv(channel * 4, channel * 2)
        self.dconv_up1 = masked_double_conv(channel * 2, channel)
        
        self.conv_last = nn.Sequential(
            MaskedConv2d_3x3(channel, channel, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 3, 1, 1),
            nn.Sigmoid(),                            
        )

    def forward(self, lighting):
        # add_random_masked(lighting)
        # lighting = gauss_noise_tensor(lighting)
        fc, fu = self.encode(lighting)
        concat_fc = fc.reshape(-1, 6 * 256, 28, 28)

        # fused_fc = self.fuse_fc(concat_fc)
        fused_fc = self.fusion_fc(concat_fc)

        fused_fc = fused_fc.unsqueeze(1).repeat(1, 6, 1, 1, 1)
        fused_fc = fused_fc.reshape(-1, 256, 28, 28)
        atten_fu = self.cross_atten(fu.reshape(-1, 6, 256, 28, 28))
        fused_feature = fused_fc + atten_fu
            
        x = self.fusion(fused_feature)
        out = self.decode(x)
        return out

    def encode(self, lighting):
        conv1 = self.dconv_down1(lighting.to(self.device))
        
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
    
        x = self.dconv_down4(x)

        # if self.training:
        #     x = add_jitter(x, 20, 1)
    
        fc = self.common_MLP(x)
        fu = self.unique_MLP(x)
        return fc, fu

    def decode(self, x):
        x = self.upsample(x)        
        
        x = self.dconv_up3(x)
        x = self.upsample(x)              

        x = self.dconv_up2(x)
        x = self.upsample(x)
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        return out

    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False

# V2
# class Masked_ConvAE(nn.Module):
#     def __init__(self, device, channel=32):
#         super(Masked_ConvAE, self).__init__()
#         self.device = device
#         self.image_chennels = 3
#         self.img_size = 224
#         self.dconv_down1 = masked_double_conv(3, channel)
#         self.dconv_down2 = masked_double_conv(channel, channel * 2)
#         self.dconv_down3 = masked_double_conv(channel * 2, channel * 4)
#         self.dconv_down4 = masked_double_conv(channel * 4, channel * 8)        
        
#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

#         self.common_MLP = nn.Sequential(
#             MaskedConv2d_3x3(channel * 8, channel * 8, padding=1),
#             nn.BatchNorm2d(channel * 8),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             MaskedConv2d_3x3(channel * 8, channel * 8, padding=1),
#             nn.BatchNorm2d(channel * 8),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#         )
        
#         self.fuse_fc = nn.Sequential(
#             nn.Conv2d(channel * 8 * 6, channel * 8, 1, 1),
#             nn.BatchNorm2d(channel * 8),
#         )

#         self.unique_MLP = nn.Sequential(
#             MaskedConv2d_3x3(channel * 8, channel * 8, padding=1),
#             nn.BatchNorm2d(channel * 8),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             MaskedConv2d_3x3(channel * 8, channel * 8, padding=1),
#             nn.BatchNorm2d(channel * 8),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#         )

#         self.cross_atten = CrossAttention()

#         self.fuse_both = iAFF(channel * 8, 4)

#         self.dconv_up3 = masked_double_conv(channel * 8, channel * 4)
#         self.dconv_up2 = masked_double_conv(channel * 4, channel * 2)
#         self.dconv_up1 = masked_double_conv(channel * 2, channel)
        
#         self.conv_last = nn.Sequential(
#             MaskedConv2d_3x3(channel, channel, padding=1),
#             nn.BatchNorm2d(channel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel, 3, 1, 1),
#             nn.Sigmoid(),                            
#         )

#     def forward(self, lighting):
#         # add_random_masked(lighting)
#         # lighting = gauss_noise_tensor(lighting)
#         x = self.encode(lighting)

#         if self.training:
#             x = add_jitter(x, 20, 1)

#         fc = self.common_MLP(x)
#         fu = self.unique_MLP(x)

#         concat_fc = fc.reshape(-1, 6 * 256, 28, 28)
#         fused_fc = self.fuse_fc(concat_fc)
#         fused_fc = fused_fc.unsqueeze(1).repeat(1, 6, 1, 1, 1)
#         fused_fc = fused_fc.reshape(-1, 256, 28, 28)

#         atten_fu = self.cross_atten(fu.reshape(-1, 6, 256, 28, 28))
#         x = self.fuse_both(fused_fc, atten_fu)
#         out = self.decode(x)
#         return out

#     def encode(self, x):
#         conv1 = self.dconv_down1(x)
        
#         x = self.maxpool(conv1)

#         conv2 = self.dconv_down2(x)
#         x = self.maxpool(conv2)
        
#         conv3 = self.dconv_down3(x)
#         x = self.maxpool(conv3)   
    
#         x = self.dconv_down4(x)

        
#         return x

#     def decode(self, x):
#         x = self.upsample(x)        
        
#         x = self.dconv_up3(x)
#         x = self.upsample(x)              

#         x = self.dconv_up2(x)
#         x = self.upsample(x)
        
#         x = self.dconv_up1(x)
        
#         out = self.conv_last(x)
#         return out

#     def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False