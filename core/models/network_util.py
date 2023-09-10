import torch
import torch.nn as nn
import random
import cv2
import skimage.exposure
import numpy as np
from numpy.random import default_rng


def INF(B,H,W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class MaskedConv2d_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, bias=True):
        super(MaskedConv2d_3x3, self).__init__()
        kernel_size = 3 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv.weight.data[:, :, 1, 1] = 0
        self.conv.weight.data[:, :, 1, 1].requires_grad = False

    def forward(self, x):
        conv_result = self.conv(x)        
        return conv_result

class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=256, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, fc, fu):
        xa = fc + fu
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = fc * wei + fu * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = fc * wei2 + fu * (1 - wei2)
        xo = self.fusion(xo)
        return xo

class CrossAttention(nn.Module):
    def __init__(self, embed_size = 256, num_heads = 4, window_size = 3):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size * 6, embed_size)
        self.value = nn.Linear(embed_size * 6, embed_size)
        self.softmax = nn.Softmax(dim=1)
        self.out_linear = nn.Linear(embed_size, embed_size)
        self.window_size = window_size

    def forward(self, x):
        B, V, C, H, W = x.size()
        seq_len = H * W * V
        q = x.permute(0, 1, 3, 4, 2).reshape(B, seq_len, C)
        k = x.permute(0, 3, 4, 1, 2).reshape(B, H*W, V*C)
        v = k
        # Linearly transform the queries, keys, and values
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        
        # Split the queries, keys, and values into multiple heads
        q = q.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute the scaled dot-product attention
        scaled_attention_logits = torch.matmul(q, k.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        mask = torch.zeros(seq_len, H*W).to(x.device)
        for i in range(seq_len):
            mask[i, max(0, i - self.window_size):min(seq_len, i + self.window_size + 1)] = 1.0
        
        # Apply the mask (if provided)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)
        
        # Concatenate and linearly transform the output
        output = output.permute(0, 2, 1, 3).contiguous().view(B, -1, C)
        output = self.out_linear(output)
        output = x.reshape(B * V, C, H, W) + output.reshape(B * V, C, H, W)
        return output

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

def masked_double_conv(in_channels, out_channels, norm=None):
    if norm == None:
        return nn.Sequential(
            MaskedConv2d_3x3(in_channels, out_channels, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            MaskedConv2d_3x3(out_channels, out_channels, padding=1),
            nn.BatchNorm2d(out_channels),
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

def gauss_noise_tensor(img, max_range = 0.4):
    assert isinstance(img, torch.Tensor)
    
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)

    scalar = torch.rand(img.shape[0]) * (max_range - 0.0) + 0.0
    scalar = scalar.reshape(-1, 1, 1, 1).to(img.device)
    out = img + scalar * torch.randn_like(img).clamp(-1, 1).to(img.device)
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