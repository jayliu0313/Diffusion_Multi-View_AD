import torch
import torch.nn as nn
from core.models.network_util import *


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
