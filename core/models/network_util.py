import torch
import torch.nn as nn

class MaskedConv2d_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, bias=False):
        super(MaskedConv2d_3x3, self).__init__()
        kernel_size = 3 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv.weight.data[:, :, 1, 1] = 0
        self.conv.weight.data[:, :, 1, 1].requires_grad = False

    def forward(self, x):
        conv_result = self.conv(x)        
        return conv_result

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
            nn.ReLU(inplace=True),
            MaskedConv2d_3x3(out_channels, out_channels, padding=1),
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