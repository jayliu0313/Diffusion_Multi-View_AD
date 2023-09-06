import torch.nn as nn
import torch
import random
from core.models.network_util import double_conv, masked_double_conv, MaskedConv2d_3x3

def add_random_masked(batch_images, scale = 0.3, prob = 0.8):
        out_images = []
        for image in batch_images:
            if random.random() <= prob:
                num = random.randint(5, 20)
                for i in range(num):
                    dim_channel, w, h = image.shape
                    x = random.randint(0, w - 1)
                    y = random.randint(0, h - 1)
                    new_w = random.randint(1, min(80, w - x))
                    new_h = random.randint(1, min(80, h - y))
                    noise = torch.randn((dim_channel, new_w, new_h)).to('cuda') * scale
                    image[:, x:x+new_w, y:y+new_h] += noise
            out_images.append(image)
        out_images = torch.stack(out_images)
        return out_images

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

    def get_mean_fc(self, six_fc):
        fc = six_fc.reshape(-1, 6, 256, 28, 28)
        mean_fc = torch.mean(fc, dim = 1)
        mean_fc = mean_fc.reshape(-1, 256, 28, 28)
        return mean_fc
    
    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False

class Convolution_AE_v2(nn.Module):

    def __init__(self, device, channel = 32):
        super(Convolution_AE_v2, self).__init__()
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
            nn.InstanceNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 8, channel * 8, 1, 1),
        )

        self.unique_MLP = nn.Sequential(
            nn.Conv2d(channel * 8, channel * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 8, channel * 8, 1, 1),
        )

        self.fusion = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 8, channel * 8, 1, 1),
            nn.ReLU(inplace=True),
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
            MaskedConv2d_3x3(channel * 8, channel * 8, 1),
        )

        self.unique_MLP = nn.Sequential(
            MaskedConv2d_3x3(channel * 8, channel * 8, padding=1),
            nn.ReLU(inplace=True),
            MaskedConv2d_3x3(channel * 8, channel * 8, 1),
        )

        self.fusion = nn.Sequential(
            nn.ReLU(inplace=True),
            MaskedConv2d_3x3(channel * 8, channel * 8, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.dconv_up3 = masked_double_conv(channel * 8, channel * 4)
        self.dconv_up2 = masked_double_conv(channel * 4, channel * 2)
        self.dconv_up1 = masked_double_conv(channel * 2, channel)
        
        self.conv_last = nn.Sequential(
            MaskedConv2d_3x3(channel, channel, padding=1),
            nn.ReLU(inplace=True),
            MaskedConv2d_3x3(channel, self.image_chennels, padding=1),
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

class Masked_ConvAE_v2(nn.Module):
    def __init__(self, device, channel=32):
        super(Masked_ConvAE_v2, self).__init__()
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

        self.unique_MLP = nn.Sequential(
            MaskedConv2d_3x3(channel * 8, channel * 8, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 8, channel * 8, 1, 1),
        )

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
        fc, fu = self.encode(lighting)
        out = self.decode(fc, fu)
        return fc, out

    def encode(self, lighting):
       
        lighting = add_random_masked(lighting)
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

class Conv_InstanceNorm(nn.Module):

    def __init__(self, device, channel = 32):
        super(Conv_InstanceNorm, self).__init__()
        self.device = device
        self.image_chennels = 3
        self.img_size = 224
        self.dconv_down1 = double_conv(3, channel, "instanceNorm")
        self.dconv_down2 = double_conv(channel, channel * 2, "instanceNorm")
        self.dconv_down3 = double_conv(channel * 2, channel * 4, "instanceNorm")
        self.dconv_down4 = double_conv(channel * 4, channel * 8, "instanceNorm")        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

        self.common_MLP = nn.Sequential(
            nn.Conv2d(channel * 8, channel * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 8, channel * 8, 1, 1),
        )

        self.unique_MLP = nn.Sequential(
            nn.Conv2d(channel * 8, channel * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 8, channel * 8, 1, 1),
        )

        self.fusion = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 8, channel * 8, 1, 1),
            nn.ReLU(inplace=True),
        )
        
        self.dconv_up3 = double_conv(channel * 8, channel * 4, "instanceNorm")
        self.dconv_up2 = double_conv(channel * 4, channel * 2, "instanceNorm")
        self.dconv_up1 = double_conv(channel * 2, channel, "instanceNorm")
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.InstanceNorm2d(channel),
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

        