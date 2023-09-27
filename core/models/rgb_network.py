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

        self.common_MLP = nn.Conv2d(channel * 8, channel * 8, 1, 1)
        # nn.Sequential(
        #     nn.Conv2d(channel * 8, channel * 8, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channel * 8, channel * 8, 1, 1),
        # )

        self.unique_MLP = nn.Conv2d(channel * 8, channel * 8, 1, 1)
        # nn.Sequential(
        #     nn.Conv2d(channel * 8, channel * 8, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channel * 8, channel * 8, 1, 1),
            
        # )

        self.fusion = nn.Conv2d(channel * 8, channel * 8, 1, 1)
        # nn.Sequential(
        #     nn.Conv2d(channel * 8, channel * 8, 1, 1),
        #     nn.ReLU(inplace=True),
        # )
        self.relu = nn.ReLU(inplace=True)
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
        return out

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
            
        x = self.relu(self.fusion(cat_feature))

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

# V4
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
#             nn.ReLU(inplace=True),
#             MaskedConv2d_3x3(channel * 8, channel * 8, padding=1),
#         )
        
#         self.unique_MLP = nn.Sequential(
#             MaskedConv2d_3x3(channel * 8, channel * 8, padding=1),
#             nn.ReLU(inplace=True),
#             MaskedConv2d_3x3(channel * 8, channel * 8, padding=1),
#         )

#         self.atten = CrossAttention()

#         self.fuse_both = nn.Sequential(
#             nn.ReLU(inplace=True),
#             MaskedConv2d_3x3(channel * 8, channel * 8, padding=1),
#             nn.ReLU(inplace=True),
#         )

#         self.dconv_up3 = masked_double_conv(channel * 8, channel * 4)
#         self.dconv_up2 = masked_double_conv(channel * 4, channel * 2)
#         self.dconv_up1 = masked_double_conv(channel * 2, channel)
        
#         self.conv_last = nn.Sequential(
#             MaskedConv2d_3x3(channel, channel, padding=1),
#             nn.ReLU(inplace=True),
#             MaskedConv2d_3x3(channel, 3, padding=1),
#             nn.Sigmoid(),                            
#         )

#         self.feature_loss = torch.nn.MSELoss()

#     def forward(self, lighting):
#         # add_random_masked(lighting)
#         if self.training:
#             lighting = gauss_noise_tensor(lighting)

#         x = self.encode(lighting)

#         if self.training:
#             x = add_jitter(x, 30, 0.5)

#         fc = self.common_MLP(x)
#         fu = self.unique_MLP(x)
#         _, C, H ,W = fc.size()
#         fc = fc.reshape(-1, 6, C, H, W)
#         mean_fc = torch.mean(fc, dim = 1)
#         mean_fc = mean_fc.unsqueeze(1).repeat(1, 6, 1, 1, 1)
#         loss_fc = self.feature_loss(fc, mean_fc)

#         random_indices = torch.randperm(6)
#         fc = fc[:, random_indices, :, :]
#         fc = fc.reshape(-1, C, H, W)

#         atten_fu = self.atten(fu.reshape(-1, 6, 256, 28, 28))
#         x = self.fuse_both(fc + atten_fu)
        
#         out = self.decode(x)
#         return out, loss_fc

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

#     def unique_rec(self, x):
#         x = self.encode(x)
#         fc = self.common_MLP(x)
#         fu = self.unique_MLP(x)
#         atten_fu = self.atten(fu.reshape(-1, 6, 256, 28, 28))
#         x = self.fuse_both(fc + atten_fu)
#         out = self.decode(x)
#         return out
    
#     def mean_rec(self, x):
#         x = self.encode(x)
#         fc = self.common_MLP(x)
#         fu = self.unique_MLP(x)
#         _, C, H ,W = fc.size()
#         fc = fc.reshape(-1, 6, C, H, W)
#         mean_fc = torch.mean(fc, dim = 1)
#         mean_fc = mean_fc.unsqueeze(1).repeat(1, 6, 1, 1, 1)
#         atten_fu = self.atten(fu.reshape(-1, 6, 256, 28, 28))
#         x = self.fuse_both(mean_fc + atten_fu)
#         out = self.decode(x)
#         return out
    
#     def freeze_model(self):
#         for param in self.parameters():
#             param.requires_grad = False


# V3
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
            nn.Conv2d(channel * 8, channel * 8,  1, 1),
        )
        
        self.unique_MLP = nn.Sequential(
            MaskedConv2d_3x3(channel * 8, channel * 8, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 8, channel * 8,  1, 1),
        )

        self.cross_atten = CrossAttention()

        self.fuse_both = nn.Sequential(
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

        self.feature_loss = torch.nn.MSELoss()

    def forward(self, lighting):
        # add_random_masked(lighting)
        if self.training:
            lighting = gauss_noise_tensor(lighting)

        x = self.encode(lighting)

        if self.training:
            x = add_jitter(x, 30, 1)

        fc = self.common_MLP(x)
        fu = self.unique_MLP(x)
        _, C, H ,W = fc.size()
        fc = fc.reshape(-1, 6, C, H, W)
        mean_fc = torch.mean(fc, dim = 1)
        mean_fc = mean_fc.unsqueeze(1).repeat(1, 6, 1, 1, 1)
        loss_fc = self.feature_loss(fc, mean_fc)

        random_indices = torch.randperm(6)
        fc = fc[:, random_indices, :, :]
        fc = fc.reshape(-1, C, H, W)

        atten_fu = self.cross_atten(fu.reshape(-1, 6, 256, 28, 28))
        x = self.fuse_both(fc + atten_fu)
        
        out = self.decode(x)
        return out, loss_fc

    def encode(self, x):
        conv1 = self.dconv_down1(x)
        
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
    
        x = self.dconv_down4(x)

        
        return x

    def decode(self, x):
        x = self.upsample(x)        
        
        x = self.dconv_up3(x)
        x = self.upsample(x)              

        x = self.dconv_up2(x)
        x = self.upsample(x)
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        return out
      
      
    def unique_rec(self, x):
        x = self.encode(x)
        fc = self.common_MLP(x)
        fu = self.unique_MLP(x)
        atten_fu = self.cross_atten(fu.reshape(-1, 6, 256, 28, 28))
        x = self.fuse_both(fc + atten_fu)
        out = self.decode(x)
        return out
    
    def mean_rec(self, x):
        x = self.encode(x)
        fc = self.common_MLP(x)
        fu = self.unique_MLP(x)
        _, C, H ,W = fc.size()
        fc = fc.reshape(-1, 6, C, H, W)
        mean_fc = torch.mean(fc, dim = 1)
        mean_fc = mean_fc.repeat(6, 1, 1, 1)
        atten_fu = self.cross_atten(fu.reshape(-1, 6, 256, 28, 28))
        x = self.fuse_both(mean_fc + atten_fu)
        out = self.decode(x)
        return out
    
    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False
