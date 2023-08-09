import os
import glob
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np
import math

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

    def __init__(self, split, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
       
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(dataset_path, self.cls, split)
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor()])    

class TestLightings(BaseDataset):
    def __init__(self, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        super().__init__(split="test_public", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.gt_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.data_paths, self.gt_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        data_tot_paths = []
        gt_tot_paths = []

        rgb_paths = glob.glob(os.path.join(self.img_path, 'data') + "/*_image_*.png")
        tiff_paths = glob.glob(os.path.join(self.img_path, 'data') + "/*.tiff")
        normal_paths = glob.glob(os.path.join(self.img_path, 'data') + "/*_normals.png")
        gt_paths = [os.path.join(self.img_path, 'data', str(i).zfill(2)+'_colors_mask.png') for i in range(len(tiff_paths))]
        
        rgb_paths.sort()
        tiff_paths.sort()
        normal_paths.sort()
        gt_paths.sort()
        
        rgb_lighting_paths = []
        rgb_6_paths = []
        for i in range(len(rgb_paths)):
            rgb_6_paths.append(rgb_paths[i])
            if (i + 1) % 6 == 0:
                rgb_lighting_paths.append(rgb_6_paths)
                rgb_6_paths = []

        sample_paths = list(zip(rgb_lighting_paths, tiff_paths, normal_paths))
        data_tot_paths.extend(sample_paths)
        gt_tot_paths.extend(gt_paths)
        
        return data_tot_paths, gt_tot_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path, gt = self.data_paths[idx], self.gt_paths[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]

        organized_pc = read_tiff_organized_pc(tiff_path)
        pc = resize_organized_pc(organized_pc, target_height=self.size, target_width=self.size)
        # organized_pc_np = pc.squeeze().permute(1, 2, 0).numpy()
        # unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        # nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        # mask = reweight(self.size, nonzero_indices)

        images = []
        for i in range(6):
            img = Image.open(rgb_path[i]).convert('RGB')
            
            img = self.rgb_transform(img)
            # img = img * mask
            images.append(img)
        images = torch.stack(images)
        # images = gauss_noise_tensor(images)
        # images = self.augment_trans(images)
        pc = pc.clone().detach().float()

        gt = Image.open(gt).convert('L')
        if np.any(gt):
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)
            label = 1
        else:
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)
            label = 0

        return (images, pc), gt[:1], label

class TrainLightings(Dataset):
    def __init__(self, img_size=224, dataset_path='datasets/eyecandies_preprocessed'):
        self.size = img_size
        self.rgb_transform = transforms.Compose(
        [transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.BICUBIC),
         transforms.ToTensor(),
         # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        # self.augment_trans = transforms.Compose([
        #     transforms.GaussianBlur((0.1, 2))
        # ])
        self.img_path = dataset_path
        self.data_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        data_tot_paths = []
        tot_labels = []
        rgb_paths = []
        tiff_paths = []
        normal_paths = []

        for cls in eyecandies_classes():
            rgb_paths.extend(glob.glob(os.path.join(self.img_path, cls, 'train', 'data', "*_image_*.png")))
            tiff_paths.extend(glob.glob(os.path.join(self.img_path, cls, 'train', 'data') + "/*.tiff"))
            normal_paths.extend(glob.glob(os.path.join(self.img_path, cls, 'train', 'data') + "/*_normals.png"))
        rgb_paths.sort()
        tiff_paths.sort()
        normal_paths.sort()
        rgb_lighting_paths = []
        rgb_6_paths = []
        
        for i in range(len(rgb_paths)):
            rgb_6_paths.append(rgb_paths[i])
            if (i + 1) % 6 == 0:
                rgb_lighting_paths.append(rgb_6_paths)
                rgb_6_paths = []
        
        sample_paths = list(zip(rgb_lighting_paths, tiff_paths, normal_paths))  # 
        data_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return data_tot_paths, tot_labels

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path, label = self.data_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]

        # organized_pc = read_tiff_organized_pc(tiff_path)
        # pc = resize_organized_pc(organized_pc, target_height=self.size, target_width=self.size)
        # organized_pc_np = pc.squeeze().permute(1, 2, 0).numpy()
        # unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        # nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        # # weight = reweight(self.size, nonzero_indices)
        # pc = pc.clone().detach().float()

        images = []
        for i in range(6):
            img = Image.open(rgb_path[i]).convert('RGB')
            img = self.rgb_transform(img)
            images.append(img)
        images = torch.stack(images)
        
        return images, label


def test_lightings_loader(args, cls):
    dataset = TestLightings(cls, args.image_size, args.data_path)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False,
                              pin_memory=True)
    return data_loader

def train_lightings_loader(args):
    dataset = TrainLightings(args.image_size, args.data_path)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False,
                              pin_memory=True)
    return data_loader

# def pretrain_data_loader(args):
#     dataset = PreTrainTensorDataset(args.data_path)
#     data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False,
#                               pin_memory=True)
#     return data_loader

# class PreTrainTensorDataset(Dataset):
#     def __init__(self, root_path):
#         super().__init__()
#         self.root_path = root_path
#         self.all_data_path = self.load_dataset()

#     def load_dataset(self):
#         data_tot_paths = []
#         rgb_paths = glob.glob(self.root_path + "/*rgb*")
#         pc_paths = glob.glob(self.root_path + "/*xyz*")
#         rgb_paths.sort()
#         pc_paths.sort()
#         sample_paths = list(zip(rgb_paths, pc_paths))
#         data_tot_paths.extend(sample_paths)
#         return data_tot_paths
    
#     def __len__(self):
#         return len(self.all_data_path)

#     def __getitem__(self, idx):
#         rgb_tensor_path = self.all_data_path[idx][0]
#         pc_tensor_path = self.all_data_path[idx][1]
        
#         rgb_tensor = torch.load(rgb_tensor_path)
#         pc_tensor = torch.load(pc_tensor_path)
        
#         label = 0

#         return (rgb_tensor, pc_tensor), label