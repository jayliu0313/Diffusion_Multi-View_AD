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
    log_file.write(f'Score Type: {args.score_type} \n')
    log_file.write(f'Seed: {args.seed} \n\n')

    log_file.write(f'VAE CheckPoint: {args.load_vae_ckpt} \n')
    log_file.write(f'Unet CheckPoint: {args.load_unet_ckpt} \n')
    log_file.write(f'ControlNet CheckPoint: {args.load_controlnet_ckpt} \n')
    log_file.write(f'Diffusion ID: {args.diffusion_id} \n')
    log_file.write(f'Revision: {args.revision} \n\n')

    log_file.write(f'Distance Function: {args.dist_function} \n\n')
    
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

class CutPaste(object):

    def __init__(self, transform = True, type = 'binary'):
        '''
        This class creates to different augmentation CutPaste and CutPaste-Scar. Moreover, it returns augmented images
        for binary and 3 way classification

        :arg
        :transform[binary]: - if True use Color Jitter augmentations for patches
        :type[str]: options ['binary' or '3way'] - classification type
        '''
        self.type = type
        if transform:
            self.transform = transforms.ColorJitter(brightness = 0.5,
                                                      contrast = 0.5,
                                                      saturation = 0.5,
                                                      hue = 0.4)
        else:
            self.transform = None

    @staticmethod
    def crop_and_paste_patch(images, patch_w, patch_h, transform, rotation=False):
        """
        Crop patch from original image and paste it randomly on the same image.

        :image: [PIL] _ original image
        :patch_w: [int] _ width of the patch
        :patch_h: [int] _ height of the patch
        :transform: [binary] _ if True use Color Jitter augmentation
        :rotation: [binary[ _ if True randomly rotates image from (-45, 45) range

        :return: augmented image
        """

        org_w, org_h = images[0].size
        mask = None

        patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        paste_left, paste_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
        patch = images[0].crop((patch_left, patch_top, patch_right, patch_bottom))
        if transform:
            patch= transform(patch)
        if rotation:
            random_rotate = random.uniform(*rotation)
            patch = patch.convert("RGBA").rotate(random_rotate, expand=True)
            mask = patch.split()[-1]

        aug_images = []
        for image in images:
            # new location
            aug_image = image.copy()
            aug_image.paste(patch, (paste_left, paste_top), mask=mask)
            aug_images.append(aug_image)
        return aug_images

    def cutpaste(self, images, area_ratio = (0.005, 0.01), aspect_ratio = ((0.3, 1) , (1, 3.3))):
        '''
        CutPaste augmentation

        :image: [PIL] - original image
        :area_ratio: [tuple] - range for area ratio for patch
        :aspect_ratio: [tuple] -  range for aspect ratio

        :return: PIL image after CutPaste transformation
        '''
        img_area = images[0].size[0] * images[0].size[1]
        patch_area = random.uniform(*area_ratio) * img_area
        patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
        patch_w  = int(np.sqrt(patch_area*patch_aspect))
        patch_h = int(np.sqrt(patch_area/patch_aspect))
        cutpaste_images = self.crop_and_paste_patch(images, patch_w, patch_h, self.transform, rotation = False)
        return cutpaste_images

    def cutpaste_scar(self, image, width = [2,16], length = [10,25], rotation = (-45, 45)):
        '''

        :image: [PIL] - original image
        :width: [list] - range for width of patch
        :length: [list] - range for length of patch
        :rotation: [tuple] - range for rotation

        :return: PIL image after CutPaste-Scare transformation
        '''
        patch_w, patch_h = random.randint(*width), random.randint(*length)
        cutpaste_scar = self.crop_and_paste_patch(image, patch_w, patch_h, self.transform, rotation = rotation)
        return cutpaste_scar

    def __call__(self, image):
        cutpaste_images = self.cutpaste(image)
        return cutpaste_images