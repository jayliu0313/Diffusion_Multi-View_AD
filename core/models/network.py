import torch.nn as nn
from torch.nn.functional import normalize
import torch
import torchvision.models as models

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )   

def gauss_noise_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)
    
    sigma = 0.1
    
    out = img + sigma * torch.randn_like(img)
    
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return out

class Convolution_AE(nn.Module):

    def __init__(self, args, device, channel = 32):
        super(Convolution_AE, self).__init__()
        self.device = device
        self.image_size = args.image_size
        self.image_chennels = 3

        # self.feature_dim = args.common_feature_dim + args.unique_feature_dim
        # self.common_dim = args.common_feature_dim
        # self.unique_dim = args.unique_feature_dim 

        self.dconv_down1 = double_conv(3, channel)
        self.dconv_down2 = double_conv(channel, channel * 2)
        self.dconv_down3 = double_conv(channel * 2, channel * 4)
        self.dconv_down4 = double_conv(channel * 4, channel * 8)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

        self.common_MLP = nn.Conv2d(channel * 8, channel * 8, 1, 1)
        self.unique_MLP = nn.Conv2d(channel * 8, channel * 8, 1, 1)
        self.fusion = nn.Conv2d(channel * 8, channel * 8, 1, 1)
        self.ac = nn.ReLU(inplace=True)
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
            
        x = self.ac(self.fusion(cat_feature))

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

# class Autoencoder(nn.Module):

#     def __init__(self, args, device, channel = 32):
#         super(Autoencoder, self).__init__()
#         self.device = device
#         image_size = args.image_size
#         self.image_chennels = 3

#         self.feature_dim = args.common_feature_dim + args.unique_feature_dim
#         self.common_dim = args.common_feature_dim
#         self.unique_dim = args.unique_feature_dim 

#         self.dconv_down1 = double_conv(3, channel)
#         self.dconv_down2 = double_conv(channel, channel * 2)
#         self.dconv_down3 = double_conv(channel * 2, channel * 4)
#         self.dconv_down4 = double_conv(channel * 4, channel * 8)        

#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
#         self.bottleneck = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(28*28*channel*8, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, self.feature_dim),

#             nn.Linear(self.feature_dim, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 28*28*channel*8),
#             nn.Unflatten(1, (channel * 8, 28, 28)),
#         )
  
#         self.dconv_up3 = double_conv(channel * 8, channel * 4)
#         self.dconv_up2 = double_conv(channel * 4, channel * 2)
#         self.dconv_up1 = double_conv(channel * 2, channel)
        
#         self.conv_last = nn.Sequential(
#             nn.Conv2d(channel, channel, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel, self.image_chennels, 1),
#             nn.Sigmoid(),                            
#         )
       
#     def forward(self, lighting):
        
#         _in = lighting
#         conv1 = self.dconv_down1(_in)
#         x = self.maxpool(conv1)
        
#         conv2 = self.dconv_down2(x)
#         x = self.maxpool(conv2)
        
#         conv3 = self.dconv_down3(x)
#         x = self.maxpool(conv3)   

#         x = self.dconv_down4(x)

#         x = self.bottleneck(x)

#         x = self.upsample(x)        
#         x = self.dconv_up3(x)
        
#         x = self.upsample(x)          
#         x = self.dconv_up2(x)
          
#         x = self.upsample(x)
#         x = self.dconv_up1(x)
        
#         out = self.conv_last(x)
        
#         return None, out
    
#     def get_common_feature(self, lighting):
#         conv1 = self.dconv_down1(lighting.to(self.device))
#         x = self.maxpool(conv1)

#         conv2 = self.dconv_down2(x)
#         x = self.maxpool(conv2)
        
#         conv3 = self.dconv_down3(x)
#         x = self.maxpool(conv3)   
    
#         x = self.dconv_down4(x)
        
#         embedded = self.bottleneck1(x)
#         common_feature = self.common_MLP(embedded[:, 0:self.common_dim])
#         return common_feature
    
#     def get_unique_feature(self, lighting):
#         conv1 = self.dconv_down1(lighting.to(self.device))
#         x = self.maxpool(conv1)

#         conv2 = self.dconv_down2(x)
#         x = self.maxpool(conv2)
        
#         conv3 = self.dconv_down3(x)
#         x = self.maxpool(conv3)   
    
#         x = self.dconv_down4(x)
        
#         embedded = self.bottleneck1(x)
#         unique_feature = self.unique_MLP(embedded[:, self.common_dim:])
#         return unique_feature
    
#     def reconstruct(self, common, unique):
#         cat_feature = torch.cat((common, unique), 1)
            
#         x = self.bottleneck2(cat_feature)

#         x = self.upsample(x)        
        
#         x = self.dconv_up3(x)
#         x = self.upsample(x)              

#         x = self.dconv_up2(x)
#         x = self.upsample(x)
        
#         x = self.dconv_up1(x)
        
#         out = self.conv_last(x)
#         return out
    

    