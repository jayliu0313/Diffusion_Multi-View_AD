import torch
import torch.nn as nn

from einops import rearrange

'''
    This module takes in CLIP + VAE embeddings and outputs CLIP-compatible embeddings.
'''
class Adapter(nn.Module):
    def __init__(self, chkpt=None):
        super(Adapter, self).__init__()

        self.save_method_name = "adapter"

        # self.pool =  nn.MaxPool2d(2)
        # self.vae2clip = nn.Linear(1280, 768)

        self.linear = nn.Linear(50, 50) # 50 x 54 shape

        # initialize weights
        with torch.no_grad():
            self.linear.weight = nn.Parameter(torch.eye(50, 50))

        if chkpt is not None:
            pass

    def forward(self, clip):
        clip = rearrange(clip, 'b c d -> b d c')
        clip = self.linear(clip)
        clip = rearrange(clip, 'b d c -> b c d')
        return clip

