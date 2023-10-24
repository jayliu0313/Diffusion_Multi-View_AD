from core.models.network_util import *
import torch.nn as nn

# Pure Normal Conv AE
class NMap_AE(nn.Module):
    def __init__(self, device, channel=32):
        super(NMap_AE, self).__init__()
        self.device = device
        self.image_chennels = 3
        self.img_size = 224
        self.dconv_down1 = double_conv(3, channel)
        self.dconv_down2 = double_conv(channel, channel * 2)
        self.dconv_down3 = double_conv(channel * 2, channel * 4)
        self.dconv_down4 = double_conv(channel * 4, channel * 8)        
        
        self.maxpool = nn.MaxPool2d(2)
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

    def forward(self, x):
        x = self.encode(x)
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

class NMap_ResnetAE(nn.Module):
    def __init__(self, device, channel=32):
        super(NMap_ResnetAE, self).__init__()
        self.device = device
        self.image_chennels = 3
        self.img_size = 224
        self.dconv_down1 = BasicBlock(3, channel, 2, short_cut=conv1x1(3, channel, 2))
        self.dconv_down2 = BasicBlock(channel, channel*2, 2, short_cut=conv1x1(channel, channel*2, 2))
        self.dconv_down3 = BasicBlock(channel*2, channel*4, 2, short_cut=conv1x1(channel*2, channel*4, 2))
        self.dconv_down4 = BasicBlock(channel*4, channel*8, 1, short_cut=conv1x1(channel*4, channel*8, 1))        
        
        upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)        

        self.dconv_up3 = BasicBlock(channel * 8, channel * 4, upsample=upsample, 
                                    short_cut=nn.Sequential(conv1x1(channel * 8, channel * 4, stride=1), upsample))
        self.dconv_up2 = BasicBlock(channel * 4, channel * 2, upsample=upsample,
                                    short_cut=nn.Sequential(conv1x1(channel * 4, channel * 2, stride=1), upsample))
        self.dconv_up1 = BasicBlock(channel * 2, channel, upsample=upsample,
                                    short_cut=nn.Sequential(conv1x1(channel * 2, channel, stride=1), upsample))
        self.conv_last = BasicBlock(channel, 3, upsample=None, short_cut=conv1x1(channel, 3, stride=1))

    def forward(self, x):
        x = self.encode(x)
        out = self.decode(x)
        return out

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

    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False