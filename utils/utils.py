import os
import numpy as np
import random
import torch
from torchvision import transforms
from PIL import ImageFilter

def log_args(args):

    log_file = open(os.path.join(args.output_dir, "log.txt"), "a", 1)
    log_file.write(f'Method Name: {args.method_name} \n')
    log_file.write(f'Score Type: {args.score_type} \n')
    log_file.write(f'Seed: {args.seed} \n\n')

    log_file.write(f'VAE CheckPoint: {args.load_vae_ckpt} \n')
    log_file.write(f'Unet CheckPoint: {args.load_unet_ckpt} \n')
    log_file.write(f'ControlNet CheckPoint: {args.load_controlnet_ckpt} \n')
    log_file.write(f'BackBone CheckPoint: {args.load_backbone_ckpt} \n')
    log_file.write(f'Nmap CheckPoint: {args.load_nmap_ckpt_path} \n\n')

    log_file.write(f'Diffusion ID: {args.diffusion_id} \n')
    log_file.write(f'Revision: {args.revision} \n\n')

    log_file.write(f'Noise Intensity: {args.noise_intensity} \n')
    
    log_file.write(f'Memory T: {args.memory_T} \n')
    log_file.write(f'Memory t: {args.memory_t} \n')
    log_file.write(f'Test T: {args.test_T} \n')
    log_file.write(f'Test t: {args.test_t} \n')
    
    log_file.write(f'Step Size: {args.step_size} \n\n')

    log_file.write(f'Opt Max Steps: {args.opt_max_steps} \n')
    log_file.write(f'Num Opt Steps: {args.num_opt_steps} \n')
    log_file.write(f'Guidance Scale: {args.guidance_scale} \n\n')

def set_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None

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
