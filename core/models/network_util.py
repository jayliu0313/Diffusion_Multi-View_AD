import torch
import torch.nn as nn
import random
import cv2
import skimage.exposure
import numpy as np
import torch.nn.functional as F
from numpy.random import default_rng


def INF(B,H,W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Decom_Block(nn.Module):
    def __init__(self, ch, mid_ch=None):
        super(Decom_Block, self).__init__()
        if mid_ch == None:
            mid_ch = ch
        self.fc_conv = conv3x3(ch, mid_ch)
        # self.fuse_fc = conv3x3(6*ch, ch)
        
        self.fu_conv = conv3x3(ch, mid_ch)
        # self.fuse_fu = conv3x3(6*ch, ch)
        # self.activate = nn.SELU()
        self.activate = nn.GELU()
        self.fuse_conv = conv3x3(mid_ch, ch)
        
    def forward(self, x):
        fc = self.fc_conv(x)
        fu = self.fu_conv(x)
        x = self.activate(self.fuse_conv(fc+fu))
        return x
    
    def rand_forward(self, x):
        fc = self.fc_conv(x)
        fu = self.fu_conv(x)

        _, C, H ,W = fc.size()
        fc = fc.reshape(-1, 6, C, H, W)
        random_indices = torch.randperm(6)
        fc = fc[:, random_indices, :, :]
        fc = fc.reshape(-1, C, H, W)
        
        _, C, H ,W = fu.size()
        fu = fu.reshape(-1, 6, C, H, W)
        B = fu.shape[0]
        random_indices = torch.randperm(B)
        fu = fu[random_indices, :, :, :]
        fu = fu.reshape(-1, C, H, W)

        x = self.activate(self.fuse_conv(fc+fu))
        return x
    
    def meanfc_prbo_randfu_forward(self, x, prob=0.5):
        fc = self.fc_conv(x)
        fu = self.fu_conv(x)
        _, C, H, W = x.size()
        fc = fc.reshape(-1, 6, C, H, W)
        meanfc = torch.mean(fc, dim = 1)
        meanfc = torch.repeat_interleave(meanfc, 6, dim=0)
        if random.uniform(0, 1) <= prob:
            fu = fu.reshape(-1, 6, C, H, W)
            B = fu.shape[0]
            random_indices = torch.randperm(B)
            fu = fu[random_indices, :, :, :]
            fu = fu.reshape(-1, C, H, W)
        return self.activate(self.fuse_conv(meanfc+fu))
         
    def prob_rand_forward(self, x, prob=0.5):
        fc = self.fc_conv(x)
        fu = self.fu_conv(x)

        if random.uniform(0, 1) <= prob:
            _, C, H, W = x.size()
            fc = fc.reshape(-1, 6, C, H, W)
            random_indices = torch.randperm(6)
            fc = fc[:, random_indices, :, :]
            fc = fc.reshape(-1, C, H, W)
  
            fu = fu.reshape(-1, 6, C, H, W)
            B = fu.shape[0]
            random_indices = torch.randperm(B)
            fu = fu[random_indices, :, :, :]
            fu = fu.reshape(-1, C, H, W)

        x = self.activate(self.fuse_conv(fc+fu))
        return x
    
    def prob_rand_forward_meanfcloss(self, x, prob=0.5):
        fc = self.fc_conv(x)
        fu = self.fu_conv(x)
        _, C, H, W = x.size()
        temp = fc.reshape(-1, 6, C, H, W)
        meanfc = torch.mean(temp, dim = 1)
        meanfc = torch.repeat_interleave(meanfc, 6, dim=0)
        loss_fc = F.mse_loss(fc, meanfc, reduction="mean")
        if random.uniform(0, 1) <= prob:
            fc = fc.reshape(-1, 6, C, H, W)
            random_indices = torch.randperm(6)
            fc = fc[:, random_indices, :, :]
            fc = fc.reshape(-1, C, H, W)
  
            fu = fu.reshape(-1, 6, C, H, W)
            B = fu.shape[0]
            random_indices = torch.randperm(B)
            fu = fu[random_indices, :, :, :]
            fu = fu.reshape(-1, C, H, W)
            
        x = self.activate(self.fuse_conv(fc+fu))
        return x, loss_fc
    
    def fusefc_prob_randfu_forward(self, x, prob=0.5):
        _, C, H, W = x.size()
        fc = self.fc_conv(x)
        temp = fc.reshape(-1, 6 * C, H, W)
        fuse_fc = self.fuse_fc(temp)
        fuse_fc = torch.repeat_interleave(fuse_fc, 6, dim=0)

        fu = self.fu_conv(x)
        if random.uniform(0, 1) <= prob:
            fu = fu.reshape(-1, 6, C, H, W)
            B = fu.shape[0]
            random_indices = torch.randperm(B)
            fu = fu[random_indices, :, :, :]
            fu = fu.reshape(-1, C, H, W)
        x = self.activate(self.fuse_conv(fuse_fc+fu))
        return x
    
    def get_fc(self, latents):
        fc = self.fc_conv(latents)
        return fc
    
    def get_fusefc(self, latents, keep_dim=True):
        _, C, H, W = latents.size()
        fc = self.fc_conv(latents)
        temp = fc.reshape(-1, 6 * C, H, W)
        # print(fc.shape)
        fuse_fc = self.fuse_fc(temp)
        # print(fuse_fc.shape)
        if keep_dim == True:
            fuse_fc = torch.repeat_interleave(fuse_fc, 6, dim=0)
        return fuse_fc

    def get_meanfc(self, latents, keep_dim=True):
        _, C, H, W = latents.size()
        fc = self.fc_conv(latents)
        fc = fc.reshape(-1, 6, C, H, W)
        mean_fc = torch.mean(fc, dim = 1)
        print(mean_fc.shape)
        if keep_dim:
            mean_fc = torch.repeat_interleave(mean_fc, 6, dim=0)
            a = mean_fc.reshape(-1, 6, C, H, W)[0, 1, :, :, :]
            b = mean_fc.reshape(-1, 6, C, H, W)[0, 2, :, :, :]
            if torch.equal(a, b):
                print("good!!!!!!!!!!!!")
        return mean_fc
        
    def get_fu(self, latents):
        fu = self.fu_conv(latents)
        return fu
    
    def fuse_both(self, fc=None, fu=None):
        if fc is not None and fu is not None:
            return self.activate(self.fuse_conv(fc + fu))
        elif fc is not None:
            return self.activate(self.fuse_conv(fc))
        elif fu is not None:
            return self.activate(self.fuse_conv(fu))


# class Decom_Block(nn.Module):
#     def __init__(self, ch, mid_ch=None):
#         super(Decom_Block, self).__init__()
#         if mid_ch == None:
#             mid_ch = ch
#         self.fc_conv = nn.Sequential(
#             nn.Conv2d(ch, mid_ch, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(mid_ch, ch, 3, padding=1),                        
#         )
#         self.fuse_fc = nn.Sequential(
#             nn.Conv2d(ch*6, ch*3, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(ch*3, ch, 3, padding=1),                          
#         )
        
#         self.fu_conv = nn.Sequential(
#             nn.Conv2d(ch, mid_ch, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(mid_ch, ch, 3, padding=1),                          
#         )
#         # self.fuse_fu = conv3x3(6*ch, ch)
#         self.fuse_conv = nn.Sequential(
#             nn.Conv2d(ch, mid_ch, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(mid_ch, ch, 3, padding=1),
#         )
        
#     def forward(self, x):
#         fc = self.fc_conv(x)
#         fu = self.fu_conv(x)
#         x = self.fuse_conv(fc+fu)
#         return x
    
#     def rand_forward(self, x):
#         fc = self.fc_conv(x)
#         fu = self.fu_conv(x)

#         _, C, H ,W = fc.size()
#         fc = fc.reshape(-1, 6, C, H, W)
#         random_indices = torch.randperm(6)
#         fc = fc[:, random_indices, :, :]
#         fc = fc.reshape(-1, C, H, W)
        
#         _, C, H ,W = fu.size()
#         fu = fu.reshape(-1, 6, C, H, W)
#         B = fu.shape[0]
#         random_indices = torch.randperm(B)
#         fu = fu[random_indices, :, :, :]
#         fu = fu.reshape(-1, C, H, W)

#         x = self.fuse_conv(fc+fu)
#         return x
    
#     def prbo_rand_meanfc_forward(self, x, prob=0.5):
#         fc = self.fc_conv(x)
#         fu = self.fu_conv(x)
#         _, C, H, W = x.size()
#         fc = fc.reshape(-1, 6, C, H, W)
#         fc = torch.mean(fc, dim = 1)
#         fc = torch.repeat_interleave(fc, 6, dim=0)
#         if random.uniform(0, 1) <= prob:
#             fu = fu.reshape(-1, 6, C, H, W)
#             B = fu.shape[0]
#             random_indices = torch.randperm(B)
#             fu = fu[random_indices, :, :, :]
#             fu = fu.reshape(-1, C, H, W)
#         return self.fuse_conv(fc+fu)
         
#     def prob_rand_forward(self, x, prob=0.5):
#         fc = self.fc_conv(x)
#         fu = self.fu_conv(x)

#         if random.uniform(0, 1) <= prob:
#             _, C, H, W = x.size()
#             fc = fc.reshape(-1, 6, C, H, W)
#             random_indices = torch.randperm(6)
#             fc = fc[:, random_indices, :, :]
#             fc = fc.reshape(-1, C, H, W)
  
#             fu = fu.reshape(-1, 6, C, H, W)
#             B = fu.shape[0]
#             random_indices = torch.randperm(B)
#             fu = fu[random_indices, :, :, :]
#             fu = fu.reshape(-1, C, H, W)

#         x = self.fuse_conv(fc+fu)
#         return x
    
#     def prob_rand_forward_meanfcloss(self, x, prob=0.5):
#         fc = self.fc_conv(x)
#         fu = self.fu_conv(x)
#         _, C, H, W = x.size()
#         temp = fc.reshape(-1, 6, C, H, W)
#         meanfc = torch.mean(temp, dim = 1)
#         mean_fc = torch.repeat_interleave(mean_fc, 6, dim=0)
#         loss_fc = F.mse_loss(fc, meanfc, reduction="mean")
#         if random.uniform(0, 1) <= prob:
#             fc = fc.reshape(-1, 6, C, H, W)
#             random_indices = torch.randperm(6)
#             fc = fc[:, random_indices, :, :]
#             fc = fc.reshape(-1, C, H, W)
  
#             fu = fu.reshape(-1, 6, C, H, W)
#             B = fu.shape[0]
#             random_indices = torch.randperm(B)
#             fu = fu[random_indices, :, :, :]
#             fu = fu.reshape(-1, C, H, W)
            
#         x = self.fuse_conv(fc+fu)
#         return x, loss_fc
    
#     def fusefc_prob_forward(self, x, prob=0.5):
#         _, C, H, W = x.size()
#         fc = self.fc_conv(x)
#         # print(fc.shape)
#         temp = fc.reshape(-1, 6 * C, H, W)
#         # print(fc.shape)
#         fuse_fc = self.fuse_fc(temp)
#         # print(fuse_fc.shape)
#         fuse_fc = fuse_fc.repeat(6, 1, 1, 1)
#         # print(fuse_fc.shape)
#         fu = self.fu_conv(x)
#         if random.uniform(0, 1) <= prob:
#             fu = fu.reshape(-1, 6, C, H, W)
#             B = fu.shape[0]
#             random_indices = torch.randperm(B)
#             fu = fu[random_indices, :, :, :]
#             fu = fu.reshape(-1, C, H, W)
#         x = self.fuse_conv(fuse_fc+fu)
#         return x
    
#     def get_fc(self, latents):
#         fc = self.fc_conv(latents)
#         return fc
    
#     def get_fusefc(self, latents, keep_dim=True):
#         _, C, H, W = latents.size()
#         fc = self.fc_conv(latents)
#         temp = fc.reshape(-1, 6 * C, H, W)
#         # print(fc.shape)
#         fuse_fc = self.fuse_fc(temp)
#         # print(fuse_fc.shape)
#         if keep_dim == True:
#             fuse_fc = torch.repeat_interleave(fuse_fc, 6, dim=0)
#         return fuse_fc

#     def get_meanfc(self, latents, keep_dim=True):
#         _, C, H, W = latents.size()
#         fc = self.fc_conv(latents)
#         fc = fc.reshape(-1, 6, C, H, W)
#         mean_fc = torch.mean(fc, dim = 1)
#         if keep_dim:
#             mean_fc = torch.repeat_interleave(mean_fc, 6, dim=0)
#         return mean_fc
        
#     def get_fu(self, latents):
#         fu = self.fu_conv(latents)
#         return fu
    
#     def fuse_both(self, fc=None, fu=None):
#         if fc is not None and fu is not None:
#             return self.activate(self.fuse_conv(fc + fu))
#         elif fc is not None:
#             return self.activate(self.fuse_conv(fc))
#         elif fu is not None:
#             return self.activate(self.fuse_conv(fu))
   
class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        short_cut = None,
        upsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
    ):
        super(BasicBlock, self).__init__()
        self.upsample = upsample
        self.short_cut = short_cut
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = norm_layer(planes)
        self.stride = stride
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        
        if self.upsample is not None:
            out = self.upsample(out)
        out = self.conv2(out)
        # out = self.bn2(out)

        if self.short_cut is not None:
            identity = self.short_cut(x)
        
        out += identity
        out = self.relu(out)

        return out

class Conv_Basic(nn.Module):
    def __init__(self, in_channel=3, out_channel=32, last=False):
        super(Conv_Basic, self).__init__()
        self.last = last
        self.dconv_down1 = double_conv(in_channel, out_channel)
        self.maxpool = nn.MaxPool2d(2)
        self.fc_conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
        )
        self.fu_conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
        )
        self.fuse_both = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel // 2, out_channel, 3, padding=1),
            nn.ReLU(inplace=True),
        ) 
        
    def forward(self, x):
        x = self.dconv_down1(x)
        fc = self.fc_conv(x)
        fu = self.fu_conv(x)
        x = self.fuse_both(fc+fu)
        if self.last != True:
            x = self.maxpool(x)
        return x
    
    def rand_fcfu(self, x):
        x = self.dconv_down1(x)
  
        fc = self.fc_conv(x)
        fu = self.fu_conv(x)

        _, C, H ,W = fc.size()
        fc = fc.reshape(-1, 6, C, H, W)
        random_indices = torch.randperm(6)
        fc = fc[:, random_indices, :, :]
        fc = fc.reshape(-1, C, H, W)
        
        _, C, H ,W = fu.size()
        fu = fu.reshape(-1, 6, C, H, W)
        B = fu.shape[0]
        random_indices = torch.randperm(B)
        fu = fu[random_indices, :, :, :]
        fu = fu.reshape(-1, C, H, W)

        x = self.fuse_both(fc + fu)
        if self.last != True:
            x = self.maxpool(x)
        return x
        

def double_conv(in_channels, out_channels, norm=None):
    if norm == None:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )   
    elif norm == "instanceNorm":
         return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )   

   
def gauss_noise_tensor(img, max_range = 1.5):
    assert isinstance(img, torch.Tensor)
    
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)
    img_norm = (
            img.norm(dim=1).unsqueeze(1) / 3
        )
    scalar = torch.rand(img.shape[0]) * (max_range - 0.0) + 0.0
    scalar = scalar.reshape(-1, 1, 1, 1).to(img.device)
    out = img + scalar * torch.randn_like(img).to(img.device) * img_norm
    out = out.clamp(0, 1)
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return out

def add_random_masked(scale = 0.5, prob = 0.8):
    # read input image
    
    # B, C, H, W = batch_images.size()
    img = cv2.imread("/mnt/home_6T/public/jayliu0313/datasets/Eyecandies/GummyBear/train/data/996_image_0.png")
    height, width = img.shape[:2]

    # define random seed to change the pattern
    seedval = 75
    rng = default_rng(seed=seedval)

    # create random noise image
    noise = rng.integers(0, 255, (height, width), np.uint8, True)

    # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)

    # stretch the blurred image to full dynamic range
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)

    # threshold stretched image to control the size
    thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]

    # apply morphology open and close to smooth out and make 3 channels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.merge([mask,mask,mask])

    # add mask to input
    result1 = cv2.add(img, mask)

    # use canny edge detection on mask
    edges = cv2.Canny(mask,50,255)

    # thicken edges and make 3 channel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
    edges = cv2.merge([edges,edges,edges])

    # merge edges with result1 (make black in result where edges are white)
    result2 = result1.copy()
    result2[np.where((edges == [255,255,255]).all(axis=2))] = [0,0,0]

    # add noise to result where mask is white
    noise = cv2.merge([noise,noise,noise])
    result3 = result2.copy()
    result3 = np.where(mask==(255,255,255), noise, result3)

    # save result
    cv2.imwrite('lena_random_blobs1.jpg', result1)
    cv2.imwrite('lena_random_blobs2.jpg', result2)
    cv2.imwrite('lena_random_blobs3.jpg', result3)

    # show results
    cv2.imshow('noise', noise)
    cv2.imshow('blur', blur)
    cv2.imshow('stretch', stretch)
    cv2.imshow('thresh', thresh)
    cv2.imshow('mask', mask)
    cv2.imshow('edges', edges)
    cv2.imshow('result1', result1)
    cv2.imshow('result2', result2)
    cv2.imshow('result3', result3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def add_jitter(feature_tokens, scale, prob):
    B, D, H, W = feature_tokens.shape
    feature_tokens = feature_tokens.reshape(B, D, H*W)
    if random.uniform(0, 1) <= prob:
        batch_size, dim_channel, num_tokens = feature_tokens.shape
        feature_norms = (
            feature_tokens.norm(dim=1).unsqueeze(1) / dim_channel
        )  # B x 1 X (H x W)
        jitter = torch.randn((batch_size, dim_channel, num_tokens)).to(feature_tokens.device)
        jitter = jitter * feature_norms * scale
        feature_tokens = feature_tokens + jitter
    return feature_tokens.reshape(B, D, H, W)
