import torch.nn as nn
import torch

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )   

class Normal_AE(nn.Module):

    def __init__(self, args, device, channel = 32):
        super(Normal_AE, self).__init__()
        self.device = device
        self.image_size = args.image_size
        self.in_chennels = 3

        self.dconv_down1 = double_conv(self.in_chennels, channel)
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
            nn.Conv2d(channel, self.in_chennels, 1),
            nn.Sigmoid(),                            
        )
       
    def forward(self, normal_map):
        embedded= self.encode(normal_map)
        out = self.decode(embedded)
        return out

    def encode(self, normal_map):
        conv1 = self.dconv_down1(normal_map.to(self.device))
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
    

    