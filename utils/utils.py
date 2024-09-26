import os
import numpy as np
import random
import torch
import torch.nn.functional as nnf
from torchvision import transforms
from PIL import ImageFilter

def log_args(args):

    log_file = open(os.path.join(args.output_dir, "log.txt"), "a", 1)
    log_file.write(f'Method Name: {args.method_name} \n')
    log_file.write(f'Reweight: {args.reweight} \n')
    log_file.write(f'Score Type: {args.score_type} \n')
    log_file.write(f'Seed: {args.seed} \n\n')

    log_file.write(f'VAE CheckPoint: {args.load_vae_ckpt} \n')
    log_file.write(f'Unet CheckPoint: {args.load_unet_ckpt} \n')
    log_file.write(f'ControlNet CheckPoint: {args.load_controlnet_ckpt} \n')
    log_file.write(f'Diffusion ID: {args.diffusion_id} \n')
    log_file.write(f'Revision: {args.revision} \n\n')

    log_file.write(f'Feature Layer: {args.feature_layers} \n')
    log_file.write(f'Top K: {args.topk} \n')
    log_file.write(f'Noise Intensity: {args.noise_intensity} \n')
    log_file.write(f'Step Size {args.step_size} \n')
    # log_file.write(f'Multi Timesteps: {args.multi_timesteps} \n')

def set_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None

def nxn_cos_sim(A, B, dim=1):
    a_norm = nnf.normalize(A, p=2, dim=dim)
    b_norm = nnf.normalize(B, p=2, dim=dim)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


class KNNGaussianBlur(torch.nn.Module):
    def __init__(self, radius : int = 4):
        super().__init__()
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=4)

    def __call__(self, img):
        map_max = img.max()
        x = self.unload(img[0] / map_max).filter(self.blur_kernel)
        final_map = self.load(x)* map_max
        return final_map