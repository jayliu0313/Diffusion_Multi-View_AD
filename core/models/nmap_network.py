from core.models.network_util import *
import torch.nn as nn

# Pure Normal Conv AE
# class NMap_AE(nn.Module):
#     def __init__(self, device, channel=32):
#         super(NMap_AE, self).__init__()
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
#             nn.Conv2d(channel, 3, 1, 1),
#             nn.Sigmoid(),                            
#         )

#     def forward(self, x):
#         if self.training:
#             x = gauss_noise_tensor(x, 1.5)
#         x = self.encode(x)
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

#     def freeze_model(self):
#         for param in self.parameters():
#             param.requires_grad = False

class NMap_Repair_Feat_AE(nn.Module):
    def __init__(self, device, channel=32):
        super(NMap_Repair_Feat_AE, self).__init__()
        self.device = device
        self.image_chennels = 3
        self.img_size = 224
        self.dconv_down1 = double_conv(3, channel)
        self.dconv_down2 = double_conv(channel, channel * 2)
        self.dconv_down3 = double_conv(channel * 2, channel * 4)
        self.dconv_down4 = double_conv(channel * 4, channel * 8)        
        
        self.Masked_Conv1 = masked5x5_3x3_double_conv(channel * 8, channel * 4)
        self.Masked_Conv2 = masked5x5_3x3_double_conv(channel * 4, channel * 8)
        self.Masked_Conv3 = masked5x5_3x3_double_conv(channel * 8, channel * 4)
        self.Masked_Conv4 = masked5x5_3x3_double_conv(channel * 4, channel * 8)
        
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
        self.feat_loss = nn.MSELoss()

    def forward(self, x):
        # if self.training:
        #     x = gauss_noise_tensor(x)
        x = self.encode(x)
        if self.training:
            broke_x = add_jitter(x, 400, 1)
        else:
            broke_x = x
            
        repair_x = self.Repair_Feat(broke_x)
        feat_loss = self.feat_loss(repair_x, x)
        out = self.decode(repair_x)
        return out
    
    def Repair_Feat(self, x):
        x = self.Masked_Conv1(x)
        x = self.Masked_Conv2(x)
        x = self.Masked_Conv3(x)
        x = self.Masked_Conv4(x)
        return x
    
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


# V6 Masked 7*7 5*5 3*3 encoder, Conv decoder
class NMap_AE(nn.Module):
    def __init__(self, device, channel=32):
        super(NMap_AE, self).__init__()
        self.device = device
        self.image_chennels = 3
        self.img_size = 224
        self.dconv_down1 = masked7x7_5x5_double_conv(3, channel)
        self.dconv_down2 = masked7x7_5x5_double_conv(channel, channel * 2)
        self.dconv_down3 = masked5x5_3x3_double_conv(channel * 2, channel * 4)
        self.dconv_down4 = masked5x5_3x3_double_conv(channel * 4, channel * 8)        
        
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
        if self.training:
            x = gauss_noise_tensor(x, 1.5)
        x = self.encode(x)
        if self.training:
            x = add_jitter(x, 100, 0.5)
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


# V5 Pure Masked 5*5 encoder decoder
# class NMap_AE(nn.Module):
#     def __init__(self, device, channel=32):
#         super(NMap_AE, self).__init__()
#         self.device = device
#         self.image_chennels = 3
#         self.img_size = 224
#         self.dconv_down1 = pure_masked5x5_double_conv(3, channel)
#         self.dconv_down2 = pure_masked5x5_double_conv(channel, channel * 2)
#         self.dconv_down3 = pure_masked5x5_double_conv(channel * 2, channel * 4)
#         self.dconv_down4 = pure_masked5x5_double_conv(channel * 4, channel * 8)        
        
#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

#         self.dconv_up3 = pure_masked5x5_double_conv(channel * 8, channel * 4)
#         self.dconv_up2 = pure_masked5x5_double_conv(channel * 4, channel * 2)
#         self.dconv_up1 = pure_masked5x5_double_conv(channel * 2, channel)
        
#         self.conv_last = nn.Sequential(
#             MaskedConv2d_5x5(channel, channel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel, 3, 1, 1),
#             nn.Sigmoid(),                            
#         )

#     def forward(self, x):
#         if self.training:
#             x = gauss_noise_tensor(x)
#         x = self.encode(x)
#         if self.training:
#             x = add_jitter(x, 30, 1)
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
#         for param in self.parameters():
#             param.requires_grad = False



# V4 Pure Masked 5*5 encoder, normal Conv decoder 
# class NMap_AE(nn.Module):
#     def __init__(self, device, channel=32):
#         super(NMap_AE, self).__init__()
#         self.device = device
#         self.image_chennels = 3
#         self.img_size = 224
#         self.dconv_down1 = pure_masked5x5_double_conv(3, channel)
#         self.dconv_down2 = pure_masked5x5_double_conv(channel, channel * 2)
#         self.dconv_down3 = pure_masked5x5_double_conv(channel * 2, channel * 4)
#         self.dconv_down4 = pure_masked5x5_double_conv(channel * 4, channel * 8)        
        
#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

#         self.dconv_up3 = double_conv(channel * 8, channel * 4)
#         self.dconv_up2 = double_conv(channel * 4, channel * 2)
#         self.dconv_up1 = double_conv(channel * 2, channel)
        
#         self.conv_last = nn.Sequential(
#             nn.Conv2d(channel, channel, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel, 3, 1, 1),
#             nn.Sigmoid(),                            
#         )

#     def forward(self, x):
#         if self.training:
#             x = gauss_noise_tensor(x)
#         x = self.encode(x)
#         if self.training:
#             x = add_jitter(x, 30, 1)
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
#         for param in self.parameters():
#             param.requires_grad = False

# V3 交錯masked 5*5, masked 3*3
# class NMap_AE(nn.Module):
#     def __init__(self, device, channel=32):
#         super(NMap_AE, self).__init__()
#         self.device = device
#         self.image_chennels = 3
#         self.img_size = 224
#         self.dconv_down1 = masked5x5_3x3_double_conv(3, channel)
#         self.dconv_down2 = masked5x5_3x3_double_conv(channel, channel * 2)
#         self.dconv_down3 = masked5x5_3x3_double_conv(channel * 2, channel * 4)
#         self.dconv_down4 = masked5x5_3x3_double_conv(channel * 4, channel * 8)        
        
#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

#         self.dconv_up3 = masked5x5_3x3_double_conv(channel * 8, channel * 4)
#         self.dconv_up2 = masked5x5_3x3_double_conv(channel * 4, channel * 2)
#         self.dconv_up1 = masked5x5_3x3_double_conv(channel * 2, channel)
        
#         self.conv_last = nn.Sequential(
#             MaskedConv2d_3x3(channel, channel, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel, 3, 1, 1),
#             nn.Sigmoid(),                            
#         )

#     def forward(self, x):
#         if self.training:
#             x = gauss_noise_tensor(x)
#         x = self.encode(x)
#         if self.training:
#             x = add_jitter(x, 30, 1)
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
#         for param in self.parameters():
#             param.requires_grad = False

# V2 第一層有5x5 masked kernel
# class NMap_AE(nn.Module):
#     def __init__(self, device, channel=32):
#         super(NMap_AE, self).__init__()
#         self.device = device
#         self.image_chennels = 3
#         self.img_size = 224
#         self.dconv_down1 = masked5x5_3x3_double_conv(3, channel)
#         self.dconv_down2 = masked_double_conv(channel, channel * 2)
#         self.dconv_down3 = masked_double_conv(channel * 2, channel * 4)
#         self.dconv_down4 = masked_double_conv(channel * 4, channel * 8)        
        
#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

#         self.dconv_up3 = masked_double_conv(channel * 8, channel * 4)
#         self.dconv_up2 = masked_double_conv(channel * 4, channel * 2)
#         self.dconv_up1 = masked_double_conv(channel * 2, channel)
        
#         self.conv_last = nn.Sequential(
#             MaskedConv2d_3x3(channel, channel, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel, 3, 1, 1),
#             nn.Sigmoid(),                            
#         )

#     def forward(self, x):
#         if self.training:
#             x = gauss_noise_tensor(x)
#         x = self.encode(x)
#         if self.training:
#             x = add_jitter(x, 30, 1)
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
#         for param in self.parameters():
#             param.requires_grad = False

# V1
# class NMap_AE(nn.Module):
#     def __init__(self, device, channel=32):
#         super(NMap_AE, self).__init__()
#         self.device = device
#         self.image_chennels = 3
#         self.img_size = 224
#         self.dconv_down1 = masked_double_conv(3, channel)
#         self.dconv_down2 = masked_double_conv(channel, channel * 2)
#         self.dconv_down3 = masked_double_conv(channel * 2, channel * 4)
#         self.dconv_down4 = masked_double_conv(channel * 4, channel * 8)        
        
#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

#         self.dconv_up3 = masked_double_conv(channel * 8, channel * 4)
#         self.dconv_up2 = masked_double_conv(channel * 4, channel * 2)
#         self.dconv_up1 = masked_double_conv(channel * 2, channel)
        
#         self.conv_last = nn.Sequential(
#             MaskedConv2d_3x3(channel, channel, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel, 3, 1, 1),
#             nn.Sigmoid(),                            
#         )

#     def forward(self, x):
#         if self.training:
#             x = gauss_noise_tensor(x)
#         x = self.encode(x)
#         if self.training:
#             x = add_jitter(x, 30, 1)
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
#         for param in self.parameters():
#             param.requires_grad = False

    