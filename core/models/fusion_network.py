import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_ch, hidden_ch=None, out_ch=None, drop=0.):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, 1, 1),
        )   

    def forward(self, x):
        return self.mlp(x)

class FeatureFusion(nn.Module):
    def __init__(self, args, rgb_dim, nmap_dim, device):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_size = args.image_size
        self.normal_dim = nmap_dim
        self.rgb_dim = rgb_dim

        self.nmap_norm = nn.LayerNorm(nmap_dim)
        self.nmap_mlp = Mlp(nmap_dim, nmap_dim, nmap_dim)

        self.rgb_norm = nn.LayerNorm(rgb_dim)
        self.rgb_mlp = Mlp(rgb_dim, rgb_dim, rgb_dim)
        
    def feature_fusion(self, rgb_feature, nmap_feature):
        rgb_feature  = self.rgb_mlp(rgb_feature)
        nmap_feature  = self.nmap_mlp(nmap_feature)
        return rgb_feature, nmap_feature

    def forward(self, rgb_feature, nmap_feature):
        rgb_feature, nmap_feature = self.feature_fusion(rgb_feature, nmap_feature)
        return rgb_feature, nmap_feature
        
    