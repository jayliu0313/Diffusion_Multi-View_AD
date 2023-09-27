from core.models.network_util import *
import torch.nn as nn

class NMap_AE(nn.Module):
    def __init__(self, device, channel=32):
        super(NMap_AE, self).__init__()
        self.device = device
        self.image_chennels = 3
        self.img_size = 224
        self.dconv_down1 = masked_double_conv(3, channel)
        self.dconv_down2 = masked_double_conv(channel, channel * 2)
        self.dconv_down3 = masked_double_conv(channel * 2, channel * 4)
        self.dconv_down4 = masked_double_conv(channel * 4, channel * 8)        
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

        self.dconv_up3 = masked_double_conv(channel * 8, channel * 4)
        self.dconv_up2 = masked_double_conv(channel * 4, channel * 2)
        self.dconv_up1 = masked_double_conv(channel * 2, channel)
        
        self.conv_last = nn.Sequential(
            MaskedConv2d_3x3(channel, channel, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 3, 1, 1),
            nn.Sigmoid(),                            
        )

    def forward(self, x):
        if self.training:
            x = gauss_noise_tensor(x)
        x = self.encode(x)
        if self.training:
            x = add_jitter(x, 30, 1)
        out = self.decode(x)
        return out

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

    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False

    