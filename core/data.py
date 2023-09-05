import os
import glob
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np
import random

def add_random_masked(image, scale = 0.1, prob = 0.5):
        if random.random() <= prob:
            num = random.randint(0, 5)
            for i in range(num):
                dim_channel, w, h = image.shape
                x = random.randint(0, w - 1)
                y = random.randint(0, h - 1)
                new_w = random.randint(1, min(45, w - x))
                new_h = random.randint(1, min(45, h - y))
                noise = torch.randn((dim_channel, new_w, new_h)) * scale
                image[:, x:x+new_w, y:y+new_h] += noise
        return image

class AddGaussianNoise(object):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        # 随机决定是否为图像添加高斯噪声
        if random.random() < 0.2:  # 这里可以调整添加噪声的概率
            # 随机选择噪声区域的位置和大小
            width, height = img.shape[2], img.shape[1]
            x = random.randint(0, width)
            y = random.randint(0, height)
            w = random.randint(1, width - x)
            h = random.randint(1, height - y)
            
            # 为选定的区域添加高斯噪声
            noise = torch.randn((img.shape[0], h, w)) * self.std + self.mean
            print(noise)
            img[:, y:y+h, x:x+w] += noise
            return img

def eyecandies_classes():
    return [
        'CandyCane',
        'ChocolateCookie',
        'ChocolatePraline',
        'Confetto',
        'GummyBear',
        'HazelnutTruffle',
        'LicoriceSandwich',
        'Lollipop',
        'Marshmallow',
        'PeppermintCandy',   
    ]

def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]

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

MEAN = [0.0, 0.0, 0.0]
STD = [255, 255, 255]

class BaseDataset(Dataset):

    def __init__(self, split, class_name, img_size, dataset_path='datasets/eyecandies'):
       
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(dataset_path, self.cls, split)
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor()])    
        self.add_masked = AddGaussianNoise(mean=0.0, std=1.0)

class TestLightings(BaseDataset):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(split="test_public", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.gt_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.data_paths, self.gt_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        data_tot_paths = []
        gt_tot_paths = []

        rgb_paths = glob.glob(os.path.join(self.img_path, 'data') + "/*_image_*.png")
        normal_paths = glob.glob(os.path.join(self.img_path, 'data') + "/*_normals.png")
        depth_paths = glob.glob(os.path.join(self.img_path, 'data') + "/*_depth.png")
        gt_paths = [os.path.join(self.img_path, 'data', str(i).zfill(2)+'_mask.png') for i in range(len(depth_paths))]
        
        rgb_paths.sort()
        normal_paths.sort()
        depth_paths.sort()
        gt_paths.sort()
        
        rgb_lighting_paths = []
        rgb_6_paths = []
        for i in range(len(rgb_paths)):
            rgb_6_paths.append(rgb_paths[i])
            if (i + 1) % 6 == 0:
                rgb_lighting_paths.append(rgb_6_paths)
                rgb_6_paths = []

        sample_paths = list(zip(rgb_lighting_paths, normal_paths, depth_paths))
        data_tot_paths.extend(sample_paths)
        gt_tot_paths.extend(gt_paths)
        
        return data_tot_paths, gt_tot_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path, gt = self.data_paths[idx], self.gt_paths[idx]
        rgb_path = img_path[0]
        normal_path = img_path[1]

        normal = Image.open(normal_path).convert('RGB')
        
        normal_map = self.rgb_transform(normal)
        images = []
        for i in range(6):
            img = Image.open(rgb_path[i]).convert('RGB')
            
            img = self.rgb_transform(img)
            
            # img = img * mask
            images.append(img)
        images = torch.stack(images)

        gt = Image.open(gt).convert('L')
        if np.any(gt):
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)
            label = 1
        else:
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)
            label = 0
        return (images, normal_map), gt[:1], label

class MemoryLightings(BaseDataset):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(split="train", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.gt_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.data_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        data_tot_paths = []

        rgb_paths = glob.glob(os.path.join(self.img_path, 'data') + "/*_image_*.png")
        normal_paths = glob.glob(os.path.join(self.img_path, 'data') + "/*_normals.png")
        depth_paths = glob.glob(os.path.join(self.img_path, 'data') + "/*_depth.png")
        
        rgb_paths.sort()
        normal_paths.sort()
        depth_paths.sort()
    
        rgb_lighting_paths = []
        rgb_6_paths = []
        for i in range(len(rgb_paths)):
            rgb_6_paths.append(rgb_paths[i])
            if (i + 1) % 6 == 0:
                rgb_lighting_paths.append(rgb_6_paths)
                rgb_6_paths = []

        sample_paths = list(zip(rgb_lighting_paths, normal_paths, depth_paths))
        data_tot_paths.extend(sample_paths)
    
        return data_tot_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path = self.data_paths[idx]
        rgb_path = img_path[0]
        normal_path = img_path[1]

        normal = Image.open(normal_path).convert('RGB')
        
        normal_map = self.rgb_transform(normal)
        images = []
        for i in range(6):
            img = Image.open(rgb_path[i]).convert('RGB')
            
            img = self.rgb_transform(img)
            # img = img * mask
            images.append(img)
        images = torch.stack(images)
        return images, normal_map

class TrainLightings(Dataset):
    def __init__(self, img_size=224, dataset_path='datasets/eyecandies_preprocessed'):
        self.size = img_size
        self.rgb_transform = transforms.Compose(
        [transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.BICUBIC),
         transforms.ToTensor(),
        ])
        self.img_path = dataset_path
        self.data_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        data_tot_paths = []
        tot_labels = []
        rgb_paths = []
        depth_paths = []
        normal_paths = []

        for cls in eyecandies_classes():
            rgb_paths.extend(glob.glob(os.path.join(self.img_path, cls, 'train', 'data', "*_image_*.png")))
            normal_paths.extend(glob.glob(os.path.join(self.img_path, cls, 'train', 'data') + "/*_normals.png"))
            depth_paths.extend(glob.glob(os.path.join(self.img_path, cls, 'train', 'data') + "/*_depth.png"))
        
        rgb_paths.sort()
        normal_paths.sort()     
        depth_paths.sort()
        
        rgb_lighting_paths = []
        rgb_6_paths = []
        
        for i in range(len(rgb_paths)):
            rgb_6_paths.append(rgb_paths[i])
            if (i + 1) % 6 == 0:
                rgb_lighting_paths.append(rgb_6_paths)
                rgb_6_paths = []
        sample_paths = list(zip(rgb_lighting_paths, normal_paths, depth_paths))
        
        data_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return data_tot_paths, tot_labels

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path, label = self.data_paths[idx], self.labels[idx]

        rgb_path = img_path[0]
        images = []
        noise_images = []
        for i in range(6):
            img = Image.open(rgb_path[i]).convert('RGB')
            img = self.rgb_transform(img)
            noise_img = add_random_masked(img)
            images.append(img)
            noise_images.append(noise_img)
        images = torch.stack(images)
        noise_images = torch.stack(noise_images)
        normal_path = img_path[1]
        depth_path = img_path[2]
        normal = Image.open(normal_path).convert('RGB')
        nmap = self.rgb_transform(normal)
        return images, noise_images, nmap

class ValLightings(Dataset):
    def __init__(self, img_size=224, dataset_path='datasets/eyecandies'):
        self.size = img_size
        self.rgb_transform = transforms.Compose(
        [transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.BICUBIC),
         transforms.ToTensor(),
        ])
        self.img_path = dataset_path
        self.data_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        data_tot_paths = []
        tot_labels = []
        rgb_paths = []
        depth_paths = []
        normal_paths = []

        for cls in eyecandies_classes():
            rgb_paths.extend(glob.glob(os.path.join(self.img_path, cls, 'val', 'data', "*_image_*.png")))
            normal_paths.extend(glob.glob(os.path.join(self.img_path, cls, 'val', 'data') + "/*_normals.png"))
            depth_paths.extend(glob.glob(os.path.join(self.img_path, cls, 'val', 'data') + "/*_depth.png"))
        
        rgb_paths.sort()
        normal_paths.sort()     
        depth_paths.sort()
    
        rgb_lighting_paths = []
        rgb_6_paths = []
        
        for i in range(len(rgb_paths)):
            rgb_6_paths.append(rgb_paths[i])
            if (i + 1) % 6 == 0:
                rgb_lighting_paths.append(rgb_6_paths)
                rgb_6_paths = []
        sample_paths = list(zip(rgb_lighting_paths, normal_paths, depth_paths))
    
        data_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return data_tot_paths, tot_labels

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path, label = self.data_paths[idx], self.labels[idx]

        rgb_path = img_path[0]
        images = []
        noise_images = []
        for i in range(6):
            img = Image.open(rgb_path[i]).convert('RGB')
            img = self.rgb_transform(img)
            noise_img = add_random_masked(img)
            images.append(img)
            noise_images.append(noise_img)
        images = torch.stack(images)
        noise_images = torch.stack(noise_images)
        normal_path = img_path[1]
        depth_path = img_path[2]
        normal = Image.open(normal_path).convert('RGB')
        nmap = self.rgb_transform(normal)
        return images, noise_images, nmap



def test_lightings_loader(args, cls, split):
    if split == 'memory':
        dataset = MemoryLightings(cls, args.image_size, args.data_path)
    elif split == 'test':
        dataset = TestLightings(cls, args.image_size, args.data_path)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False,
                              pin_memory=True)
    return data_loader

def train_lightings_loader(args):
    dataset = TrainLightings(args.image_size, args.data_path)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True,
                              pin_memory=True)
    return data_loader

def val_lightings_loader(args):
    dataset = ValLightings(args.image_size, args.data_path)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, drop_last=True,
                              pin_memory=True)
    return data_loader
