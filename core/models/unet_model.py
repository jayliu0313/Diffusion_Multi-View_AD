import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

def gauss_noise_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)
    
    sigma = 0.2
    
    out = img + sigma * torch.randn_like(img)
    
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return out

class UNet(nn.Module):

    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.image_size = args.image_size
        self.image_chennels = 3

        self.feature_dim = args.common_feature_dim + args.unique_feature_dim
        self.common_dim = args.common_feature_dim
        self.unique_dim = args.unique_feature_dim 

        self.dconv_down1 = double_conv(3, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 256)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.bottleneck1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28*256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.feature_dim),
        )

        self.bottleneck2 = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 28*28*256),
            nn.Unflatten(1, (256, 28, 28)),
        )

        self.common_MLP = nn.Linear(self.common_dim, self.common_dim)
        
        
        self.unique_MLP = nn.Linear(self.unique_dim, self.unique_dim)
        

        self.dconv_up3 = double_conv(256 + 128, 128)
        self.dconv_up2 = double_conv(128 + 64, 64)
        self.dconv_up1 = double_conv(64 + 32, 32)
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(32, self.image_chennels, 1),
            nn.Sigmoid(),
        )
 
    def forward(self, lightings):
        _in = lightings.to(self.device)
        _in = gauss_noise_tensor(_in)

        conv1 = self.dconv_down1(_in)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
    
        x = self.dconv_down4(x)
        
        embedded = self.bottleneck1(x)
        fc = self.common_MLP(embedded[:, 0:self.common_dim])
        fu = self.unique_MLP(embedded[:, self.common_dim:])
        cat_feature = torch.cat((fc, fu), 1)
        
        x = self.bottleneck2(cat_feature)

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
    
        return fc, out
    
    def reconstruct(self, lightings):
        _in = lightings.to(self.device)

        conv1 = self.dconv_down1(_in)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
    
        x = self.dconv_down4(x)
        
        embedded = self.bottleneck1(x)
        fc = self.common_MLP(embedded[:, 0:self.common_dim])
        fu = self.unique_MLP(embedded[:, self.common_dim:])
        cat_feature = torch.cat((fc, fu), 1)
        
        x = self.bottleneck2(cat_feature)

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
    
        return fc, out

    def get_feature(self, lightings):
        _in = lightings.to(self.device)
        conv1 = self.dconv_down1(_in)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
    
        x = self.dconv_down4(x)
        
        embedded = self.bottleneck1(x)
        fc = self.common_MLP(embedded[:, 0:self.common_dim])
        fu = self.unique_MLP(embedded[:, self.common_dim:])
        return fc, fu
