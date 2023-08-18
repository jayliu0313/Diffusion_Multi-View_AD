import torch
import torch.nn as nn

from core.models.network_util import double_conv

class Mlp(nn.Module):
    def __init__(self, in_ch, hidden_ch=None, out_ch=None, drop=0.):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, 1, padding=1),
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
        self.nmap_mlp = Mlp(in_features=nmap_dim, hidden_features=nmap_dim, out_ch=nmap_dim)

        self.rgb_norm = nn.LayerNorm(rgb_dim)
        self.rgb_mlp = Mlp(in_features=rgb_dim, hidden_features=rgb_dim, out_ch=rgb_dim)
        
        self.T = 1

    def feature_fusion(self, rgb_feature, nmap_feature):
        rgb_feature  = self.rgb_mlp(self.rgb_norm(rgb_feature))
        nmap_feature  = self.nmap_mlp(self.nmap_norm(nmap_feature))
        return rgb_feature, nmap_feature

    def forward(self, rgb_feature, nmap_feature):
        rgb_feature, nmap_feature = self.feature_fusion(rgb_feature, nmap_feature)
        print(rgb_feature.shape)
        print(nmap_feature.shape)
    