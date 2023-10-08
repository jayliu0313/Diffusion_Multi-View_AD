import torch.nn as nn
import torch
import random
from core.models.network_util import *
from utils.utils import KNNGaussianBlur


# v1 pureConv_with fc fu, wo atten fu
class Conv_AE(nn.Module):
    def __init__(self, device, channel=32):
        super(Conv_AE, self).__init__()
        self.device = device
        self.image_chennels = 3
        self.img_size = 224
        self.fc_dim = 256
        self.fu_dim = 64
        self.dconv_down1 = double_conv(3, channel)
        self.dconv_down2 = double_conv(channel, channel * 2)
        self.dconv_down3 = double_conv(channel * 2, channel * 4)
        self.dconv_down4 = double_conv(channel * 4, channel * 8)        
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.common_MLP = nn.Sequential(
            nn.Conv2d(channel * 8, channel * 8, 3, padding=1),
        )
        
        self.unique_MLP = nn.Sequential(
            nn.Conv2d(channel * 8, 256, 3, padding=1),
        )
        self.fuse_both = nn.Sequential(
            nn.Conv2d(256, channel * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 8, channel * 8, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dconv_up3 = double_conv(channel * 8, channel * 4)
        self.dconv_up2 = double_conv(channel * 4, channel * 2)
        self.dconv_up1 = double_conv(channel * 2, channel)
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 3, 1, 1),
            nn.Sigmoid(),                            
        )

        self.fc_loss = torch.nn.MSELoss()
        self.fu_loss = torch.nn.MSELoss()

    def forward(self, x):
        # add_random_masked(lighting)
        if self.training:
            x = gauss_noise_tensor(x, 1.0)

        x = self.encode(x)

        # if self.training:
        #     x = add_jitter(x, 30, 0.5)
        fc = self.common_MLP(x)
        fu = self.unique_MLP(x)
        
        x = self.fuse_both(fc + fu)
        
        out = self.decode(x)
        return out
    
    def rand_rec(self, x):
        # add_random_masked(lighting)
        if self.training:
            x = gauss_noise_tensor(x, 1.0)

        x = self.encode(x)

        # if self.training:
        #     x = add_jitter(x, 30, 0.5)
        fc = self.common_MLP(x)
        fu = self.unique_MLP(x)
        
        # fc process
        _, C, H ,W = fc.size()
        fc = fc.reshape(-1, 6, C, H, W)
        B = fc.shape[0]
        mean_fc = torch.mean(fc, dim = 1)
        mean_fc = mean_fc.unsqueeze(1).repeat(1, 6, 1, 1, 1)
        loss_fc = self.fc_loss(fc, mean_fc)
        random_indices = torch.randperm(6)
        fc = fc[:, random_indices, :, :]
        fc = fc.reshape(-1, C, H, W)

        # fu process
        _, C, H ,W = fu.size()
        fu = fu.reshape(-1, 6, C, H, W)
        
        mean_fu = torch.mean(fu, dim = 0)
        mean_fu = mean_fu.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        loss_fu = self.fu_loss(fu, mean_fu)
        
        random_indices = torch.randperm(B)
        fu = fu[random_indices, :, :, :]
        fu = fu.reshape(-1, C, H, W)
        
        x = self.fuse_both(fc + fu)
        out = self.decode(x)
        return out, loss_fc, loss_fu
    
    def rec_rand_rec(self, x):
        if self.training:
            x = gauss_noise_tensor(x, 1.0)
            
        x = self.encode(x)
        # if self.training:
        #     x = add_jitter(x, 30, 0.5)
        fc = self.common_MLP(x)
        fu = self.unique_MLP(x)
        
        x = self.fuse_both(fc + fu)
        out = self.decode(x)
        # fc process
        _, C, H ,W = fc.size()
        fc = fc.reshape(-1, 6, C, H, W)
        B = fc.shape[0]
        mean_fc = torch.mean(fc, dim = 1)
        mean_fc = mean_fc.unsqueeze(1).repeat(1, 6, 1, 1, 1)
        loss_fc = self.fc_loss(fc, mean_fc)
        random_indices = torch.randperm(6)
        rand_fc = fc[:, random_indices, :, :]
        rand_fc = rand_fc.reshape(-1, C, H, W)

        # fu process
        _, C, H ,W = fu.size()
        fu = fu.reshape(-1, 6, C, H, W)
        
        mean_fu = torch.mean(fu, dim = 0)
        mean_fu = mean_fu.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        loss_fu = self.fu_loss(fu, mean_fu)
        
        random_indices = torch.randperm(B)
        rand_fu = fu[random_indices, :, :, :]
        rand_fu = rand_fu.reshape(-1, C, H, W)
        
        x = self.fuse_both(rand_fc + rand_fu)
        rand_out = self.decode(x)
        
        return out, rand_out, loss_fc, loss_fu
    
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

    def rec(self, x):
        if self.training:
            x = gauss_noise_tensor(x, 1.0)
        x = self.encode(x)
        fc = self.common_MLP(x)
        fu = self.unique_MLP(x)
        x = self.fuse_both(fc + fu)
        out = self.decode(x)
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


class Conv_Decomposition_AE(nn.Module):
    def __init__(self, device, channel=32):
        super(Conv_Decomposition_AE, self).__init__()
        self.device = device
        self.image_chennels = 3
        self.img_size = 224
        # self.fc_dim = 256
        # self.fu_dim = 64
        self.dconv_down1 = Conv_Basic(3, channel)
        self.dconv_down2 = Conv_Basic(channel, channel*2)
        self.dconv_down3 = Conv_Basic(channel*2, channel*4)
        self.dconv_down4 = Conv_Basic(channel*4, channel*8, last=True)     
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
      
        self.dconv_up3 = double_conv(channel * 8, channel * 4)
        self.dconv_up2 = double_conv(channel * 4, channel * 2)
        self.dconv_up1 = double_conv(channel * 2, channel)
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 3, 1, 1),
            nn.Sigmoid(),                            
        )

        # self.fc_loss = torch.nn.MSELoss()
        # self.fu_loss = torch.nn.MSELoss()

    def forward(self, x):
        # add_random_masked(lighting)
        if self.training:
            x = gauss_noise_tensor(x, 1.0)

        x = self.encode(x)

        # if self.training:
        #     x = add_jitter(x, 30, 0.5)
        out = self.decode(x)
        return out
    
    def rand_rec(self, x):
        # add_random_masked(lighting)
        if self.training:
            x = gauss_noise_tensor(x, 1.0)
        x = self.dconv_down1.rand_fcfu(x)
        x = self.dconv_down2.rand_fcfu(x)
        x = self.dconv_down3.rand_fcfu(x)
        x = self.dconv_down4.rand_fcfu(x)
        x = self.decode(x)
        return x
    
    def rec_randrec(self, x):
        if self.training:
            x = gauss_noise_tensor(x, 1.0)
            
        _x = self.encode(x)
        out = self.decode(_x)

        x = self.dconv_down1.rand_fcfu(x)
        x = self.dconv_down2.rand_fcfu(x)
        x = self.dconv_down3.rand_fcfu(x)
        x = self.dconv_down4.rand_fcfu(x)
        
        rand_out = self.decode(x)
        
        return out, rand_out

    def encode(self, x):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        conv4 = self.dconv_down4(conv3)

        return conv4

    def decode(self, x):
        x = self.upsample(x)        
        
        x = self.dconv_up3(x)
        x = self.upsample(x)              

        x = self.dconv_up2(x)
        x = self.upsample(x)
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        return out

    def rec(self, x):
        if self.training:
            x = gauss_noise_tensor(x, 1.0)
        x = self.encode(x)
        out = self.decode(x)
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

# ResNet Layerwise Decomposition
class ResNet_Decom_AE(nn.Module):
    def __init__(self, device, channel=32):
        super(ResNet_Decom_AE, self).__init__()
        self.device = device
        self.image_chennels = 3
        self.img_size = 224
        # self.fc_dim = 256
        # self.fu_dim = 64
        
        self.dconv_down1 = BasicBlock(3, channel, 2, short_cut=conv1x1(3, channel, 2))
        self.decom_block1 = Decom_Block(channel)
        self.dconv_down2 = BasicBlock(channel, channel*2, 2, short_cut=conv1x1(channel, channel*2, 2))
        self.decom_block2 = Decom_Block(channel*2)
        self.dconv_down3 = BasicBlock(channel*2, channel*4, 2, short_cut=conv1x1(channel*2, channel*4, 2))
        self.decom_block3 = Decom_Block(channel*4)
        self.dconv_down4 = BasicBlock(channel*4, channel*8, 1, short_cut=conv1x1(channel*4, channel*8, 1))     
        self.decom_block4 = Decom_Block(channel*8)
        
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
      
        self.dconv_up3 = BasicBlock(channel * 8, channel * 4, upsample=nn.Upsample(scale_factor=2, mode="bilinear"), 
                                    short_cut=nn.Sequential(conv1x1(channel * 8, channel * 4, stride=1), nn.Upsample(scale_factor=2, mode="bilinear")))
        self.dconv_up2 = BasicBlock(channel * 4, channel * 2, upsample=nn.Upsample(scale_factor=2, mode="bilinear"),
                                    short_cut=nn.Sequential(conv1x1(channel * 4, channel * 2, stride=1), nn.Upsample(scale_factor=2, mode="bilinear")))
        self.dconv_up1 = BasicBlock(channel * 2, channel, upsample=nn.Upsample(scale_factor=2, mode="bilinear"),
                                    short_cut=nn.Sequential(conv1x1(channel * 2, channel, stride=1), nn.Upsample(scale_factor=2, mode="bilinear")))
        self.conv_last = BasicBlock(channel, 3, upsample=None, short_cut=conv1x1(channel, 3, stride=1))

    def forward(self, x):
        # add_random_masked(lighting)
        if self.training:
            x = gauss_noise_tensor(x, 1.0)

        x = self.encode(x)

        # if self.training:
        #     x = add_jitter(x, 30, 0.5)
        out = self.decode(x)
        return out
    
    def rand_forward(self, x):
        # add_random_masked(lighting)
        if self.training:
            x = gauss_noise_tensor(x, 1.0)
        x = self.rand_encode(x)
        x = self.decode(x)
        return x

    def rand_encode(self, x):
        x = self.dconv_down1(x)
        x = self.decom_block1.rand_forward(x)
        x = self.dconv_down2(x)
        x = self.decom_block2.rand_forward(x)
        x = self.dconv_down3(x)
        x = self.decom_block3.rand_forward(x)
        x = self.dconv_down4(x)
        x = self.decom_block4.rand_forward(x)
        return x
    
    def encode(self, x):
        x = self.dconv_down1(x)
        x = self.decom_block1(x)
        x = self.dconv_down2(x)
        x = self.decom_block2(x)
        x = self.dconv_down3(x)
        x = self.decom_block3(x)
        x = self.dconv_down4(x)
        x = self.decom_block4(x)
        return x

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

# ResNet BottleNeck Decomposition
class ResNetAE_Decom_Bottleneck(nn.Module):
    def __init__(self, device, channel=32):
        super(ResNetAE_Decom_Bottleneck, self).__init__()
        self.device = device
        self.image_chennels = 3
        self.img_size = 224
        # self.fc_dim = 256
        # self.fu_dim = 64
        
        self.dconv_down1 = BasicBlock(3, channel, 2, short_cut=conv1x1(3, channel, 2))
        self.dconv_down2 = BasicBlock(channel, channel*2, 2, short_cut=conv1x1(channel, channel*2, 2))
        self.dconv_down3 = BasicBlock(channel*2, channel*4, 2, short_cut=conv1x1(channel*2, channel*4, 2))
        self.dconv_down4 = BasicBlock(channel*4, channel*8, 1, short_cut=conv1x1(channel*4, channel*8, 1))     
        self.decomp_block = Decom_Block(channel*8)
      
        self.dconv_up3 = BasicBlock(channel * 8, channel * 4, upsample=nn.Upsample(scale_factor=2, mode="bilinear"), 
                                    short_cut=nn.Sequential(conv1x1(channel * 8, channel * 4, stride=1), nn.Upsample(scale_factor=2, mode="bilinear")))
        self.dconv_up2 = BasicBlock(channel * 4, channel * 2, upsample=nn.Upsample(scale_factor=2, mode="bilinear"),
                                    short_cut=nn.Sequential(conv1x1(channel * 4, channel * 2, stride=1), nn.Upsample(scale_factor=2, mode="bilinear")))
        self.dconv_up1 = BasicBlock(channel * 2, channel, upsample=nn.Upsample(scale_factor=2, mode="bilinear"),
                                    short_cut=nn.Sequential(conv1x1(channel * 2, channel, stride=1), nn.Upsample(scale_factor=2, mode="bilinear")))
        self.conv_last = BasicBlock(channel, 3, upsample=None, short_cut=conv1x1(channel, 3, stride=1))

    def forward(self, x):
        x = self.encode(x)
        x = self.decomp_block(x)
        out = self.decode(x)
        return out
    
    def rec_randrec_forward(self, x):
        # add_random_masked(lighting)
        # if self.training:
        #     x = gauss_noise_tensor(x, 1.0)
        x = self.encode(x)
        
        # normal reconstruction
        feat = self.decomp_block(x)
        out = self.decode(feat)
        # random change fc fu reconstruction
        rand_feat = self.decomp_block.rand_forward(x)
        rand_out = self.decode(rand_feat)
        return out, rand_out

    def encode(self, x):
        x = self.dconv_down1(x)
        x = self.dconv_down2(x)
        x = self.dconv_down3(x)
        x = self.dconv_down4(x)
        return x

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
        fc = self.get_fc(x)
        fu = self.get_fu(x)  
        mean_fc = torch.mean(fc, dim = 0)
        mean_fc = mean_fc.repeat(6, 1, 1, 1)
        x = self.fuse_both(mean_fc + fu)
        out = self.decode(x)
        return out
    
    def get_fc(self, x):
        fc = self.decomp_block.get_fc(x)
        return fc
        
    def get_meanfc(self, x):
        mean_fc = self.decomp_block.get_meanfc(x)
        return mean_fc
    
    def get_fu(self, x):
        x = self.decomp_block.get_fu(x)
    
    def fuse_both(self, fc, fu):
        return self.decomp_block.fuse_both(fc, fu) 
    
    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False

# pureConv_woFCFU
# class Conv_AE(nn.Module):
#     def __init__(self, device, channel=32):
#         super(Conv_AE, self).__init__()
#         self.device = device
#         self.image_chennels = 3
#         self.img_size = 224
#         self.dconv_down1 = double_conv(3, channel)
#         self.dconv_down2 = double_conv(channel, channel * 2)
#         self.dconv_down3 = double_conv(channel * 2, channel * 4)
#         self.dconv_down4 = double_conv(channel * 4, channel * 8)        
        
#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

#         self.dconv_up3 = double_conv(channel * 8, channel * 4)
#         self.dconv_up2 = double_conv(channel * 4, channel * 2)
#         self.dconv_up1 = double_conv(channel * 2, channel)
        
#         self.conv_last = nn.Sequential(
#             nn.Conv2d(channel, channel, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel, 3, 3, padding=1),
#             nn.Sigmoid(),                            
#         )

#         # self.feature_loss = torch.nn.MSELoss()

#     def forward(self, lighting):
#         # add_random_masked(lighting)
#         if self.training:
#             lighting = gauss_noise_tensor(lighting)

#         x = self.encode(lighting)

#         if self.training:
#             x = add_jitter(x, 100, 0.5)
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

#     def rec(self, x):
#         x = self.encode(x)
#         out = self.decode(x)
#         return out

#     def freeze_model(self):
#         for param in self.parameters():
#             param.requires_grad = False


# V3 pure 5x5 msked encoder, normal conv 3x3 decoder
# class Masked_ConvAE(nn.Module):
#     def __init__(self, device, channel=32):
#         super(Masked_ConvAE, self).__init__()
#         self.device = device
#         self.image_chennels = 3
#         self.img_size = 224
#         self.dconv_down1 = pure_masked5x5_double_conv(3, channel)
#         self.dconv_down2 = pure_masked5x5_double_conv(channel, channel * 2)
#         self.dconv_down3 = pure_masked5x5_double_conv(channel * 2, channel * 4)
#         self.dconv_down4 = pure_masked5x5_double_conv(channel * 4, channel * 8)        
        
#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

#         self.common_MLP = nn.Sequential(
#             MaskedConv2d_5x5(channel * 8, channel * 8),
#             nn.ReLU(inplace=True),
#             MaskedConv2d_5x5(channel * 8, channel * 8),
#         )
        
#         self.unique_MLP = nn.Sequential(
#             MaskedConv2d_5x5(channel * 8, channel * 8),
#             nn.ReLU(inplace=True),
#             MaskedConv2d_5x5(channel * 8, channel * 8),
#         )

#         self.atten = CrossAttention()

#         self.fuse_both = nn.Sequential(
#             nn.ReLU(inplace=True),
#             MaskedConv2d_5x5(channel * 8, channel * 8),
#             nn.ReLU(inplace=True),
#         )

#         self.dconv_up3 = double_conv(channel * 8, channel * 4)
#         self.dconv_up2 = double_conv(channel * 4, channel * 2)
#         self.dconv_up1 = double_conv(channel * 2, channel)
        
#         self.conv_last = nn.Sequential(
#             nn.Conv2d(channel, channel, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel, 3, 3, padding=1),
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

#     def rec(self, x):
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
#         mean_fc = mean_fc.repeat(6, 1, 1, 1)
#         atten_fu = self.atten(fu.reshape(-1, 6, 256, 28, 28))
#         x = self.fuse_both(mean_fc + atten_fu)
#         out = self.decode(x)
#         return out
    
#     def freeze_model(self):
#         for param in self.parameters():
#             param.requires_grad = False


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

#     def rec(self, x):
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
#         mean_fc = mean_fc.repeat(6, 1, 1, 1)
#         atten_fu = self.atten(fu.reshape(-1, 6, 256, 28, 28))
#         x = self.fuse_both(mean_fc + atten_fu)
#         out = self.decode(x)
#         return out
    
#     def freeze_model(self):
#         for param in self.parameters():
#             param.requires_grad = False


# V1
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
#             nn.Conv2d(channel * 8, channel * 8,  1, 1),
#         )
        
#         self.unique_MLP = nn.Sequential(
#             MaskedConv2d_3x3(channel * 8, channel * 8, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel * 8, channel * 8,  1, 1),
#         )

#         self.cross_atten = CrossAttention()

#         self.fuse_both = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel * 8, channel * 8, 1, 1),
#             nn.ReLU(inplace=True),
#         )

#         self.dconv_up3 = masked_double_conv(channel * 8, channel * 4)
#         self.dconv_up2 = masked_double_conv(channel * 4, channel * 2)
#         self.dconv_up1 = masked_double_conv(channel * 2, channel)
        
#         self.conv_last = nn.Sequential(
#             MaskedConv2d_3x3(channel, channel, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel, 3, 1, 1),
#             nn.Sigmoid(),                            
#         )

#         self.feature_loss = torch.nn.MSELoss()

#     def forward(self, lighting):
#         # add_random_masked(lighting)
#         if self.training:
#             lighting = gauss_noise_tensor(lighting)

#         x = self.encode(lighting)

#         if self.training:
#             x = add_jitter(x, 30, 1)

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

#         atten_fu = self.cross_atten(fu.reshape(-1, 6, 256, 28, 28))
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
      
      
#     def rec(self, x):
#         x = self.encode(x)
#         fc = self.common_MLP(x)
#         fu = self.unique_MLP(x)
#         atten_fu = self.cross_atten(fu.reshape(-1, 6, 256, 28, 28))
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
#         mean_fc = mean_fc.repeat(6, 1, 1, 1)
#         atten_fu = self.cross_atten(fu.reshape(-1, 6, 256, 28, 28))
#         x = self.fuse_both(mean_fc + atten_fu)
#         out = self.decode(x)
#         return out
    
#     def freeze_model(self):
#         for param in self.parameters():
#             param.requires_grad = False
