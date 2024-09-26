import os
import glob
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from utils.pair_augment import *
from torch.utils.data import DataLoader

import numpy as np

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

class BaseDataset(Dataset):

    def __init__(self, split, class_name, img_size, dataset_path='datasets/eyecandies'):
       
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(dataset_path, self.cls, split)
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor()])    

# load testing image and testing normal map    
class TestLightings(BaseDataset):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(split="test_public", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.gt_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
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
        text_prompt = "A photo of a " + self.cls
        # text_prompt = ""
        normal = Image.open(normal_path).convert('RGB')
        
        normal_map = self.rgb_transform(normal)
        images = []
        for i in range(6):
            img = Image.open(rgb_path[i]).convert('RGB')
            img = self.rgb_transform(img)
            images.append(img)
        images = torch.stack(images)
        # images = gauss_noise_tensor(images)
        gt = Image.open(gt).convert('L')
        if np.any(gt):
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)
            label = 1
        else:
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)
            label = 0
        return (images, normal_map, text_prompt), gt[:1], label

# load training data to build memory bank
class MemoryLightings(BaseDataset):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(split="train", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
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
        text_prompt = "A photo of a " + self.cls 
        # text_prompt = ""
        normal_map = Image.open(normal_path).convert('RGB')
        
        n_map = []
        n_map.append(normal_map)
        images = []
        for i in range(6):
            img = Image.open(rgb_path[i]).convert('RGB')
            images.append(img)

        aug_imgs = [self.rgb_transform(img) for img in images]
        aug_nmap = [self.rgb_transform(nmap) for nmap in n_map]
        aug_imgs = torch.stack(aug_imgs)
        aug_nmap = torch.cat(aug_nmap)
        return aug_imgs, aug_nmap, text_prompt

# load training data (6 images + 3D normal map)   
class TrainLightings(Dataset):
    def __init__(self, img_size=224, dataset_path='datasets/eyecandies_preprocessed'):
        self.size = img_size
        self.rgb_transform = transforms.Compose(
        [transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        ])
        self.cls_list = []
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
            self.cls_list.extend([cls] * 1000)

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
        cls = self.cls_list[idx]
        text_prompt = "A photo of a " + cls
        rgb_path = img_path[0]
        images = []
        # noise_images = []
        for i in range(6):
            img = Image.open(rgb_path[i]).convert('RGB')
            if not img.mode == "RGB":
                img = img.convert("RGB")
            img = self.rgb_transform(img) 
            images.append(img)
        images = torch.stack(images)
        
        normal_path = img_path[1]
        # depth_path = img_path[2]
        normal = Image.open(normal_path).convert('RGB')
        nmap = self.rgb_transform(normal)
        return images, nmap, text_prompt

# load validation data (6 images + 3D normal map)
class ValLightings(Dataset):
    def __init__(self, img_size=224, dataset_path='datasets/eyecandies'):
        self.size = img_size
        self.rgb_transform = transforms.Compose(
        [transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        ])
        self.cls_list = []
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
            self.cls_list.extend([cls] * 100)
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
        cls = self.cls_list[idx]
        text_prompt = "A photo of a " + cls
        rgb_path = img_path[0]
        images = []
        # noise_images = []
        for i in range(6):
            img = Image.open(rgb_path[i]).convert('RGB')
            img = self.rgb_transform(img)
            # noise_img = add_random_masked(img)
            images.append(img)
            # noise_images.append(noise_img)
        images = torch.stack(images)
        # noise_images = torch.stack(noise_images)
        normal_path = img_path[1]
        depth_path = img_path[2]
        normal = Image.open(normal_path).convert('RGB')
        nmap = self.rgb_transform(normal)
        return images, nmap, text_prompt

def test_lightings_loader(args, cls, split):
    if split == 'memory':
        dataset = MemoryLightings(cls, args.image_size, args.data_path)
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False,
                                pin_memory=True)
    elif split == 'test':
        dataset = TestLightings(cls, args.image_size, args.data_path)
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=args.workers, drop_last=False,
                                pin_memory=True)
    return data_loader

def train_lightings_loader(args):
    dataset = TrainLightings(args.image_size, args.data_path)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, drop_last=True,
                              pin_memory=True)
    return data_loader

def val_lightings_loader(args):
    dataset = ValLightings(args.image_size, args.data_path)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, drop_last=True,
                              pin_memory=True)
    return data_loader


######################################################
#                     MVTEC 3D-AD                    #
######################################################
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

class MVTec3DTrain(Dataset):
    def __init__(self, img_size, dataset_path, class_name=None, split='train', is_memory=True):
        self.img_size = img_size
        self.img_path = dataset_path
        self.split = split
        self.class_name = class_name
        
        if split == "train":
            if is_memory == True:
                self.paired_transform = Compose(
                [
                    Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC, interpolation_tg=transforms.InterpolationMode.BICUBIC),
                    ToTensor(),
                ])
            else:
                self.paired_transform = Compose(
                [
                    RandomVerticalFlip(p=0.5),
                    RandomHorizontalFlip(p=0.5),
                    RandomRotation(degrees=(-180, 180)),
                    Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC, interpolation_tg=transforms.InterpolationMode.BICUBIC),
                    ToTensor(),
                ])
        else:
            self.paired_transform = Compose(
            [
                Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC, interpolation_tg=transforms.InterpolationMode.BICUBIC),
                ToTensor(),
            ])
        self.cls_list = []
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []

        rgb_paths = []
        tiff_paths = []
        nmap_paths = []

        if self.class_name == None:
            cls_list = mvtec3d_classes()
        else:
            cls_list = [self.class_name]
        
        for cls in cls_list:
            rgb_path = glob.glob(os.path.join(self.img_path, cls, self.split, 'good', 'rgb') + "/*.png")
            tiff_path = glob.glob(os.path.join(self.img_path, cls, self.split, 'good', 'xyz') + "/*.tiff")
            nmap_path = glob.glob(os.path.join(self.img_path, cls, self.split, 'good', 'nmap') + "/*.png")
            self.cls_list.extend([cls] * len(rgb_path))
            rgb_paths.extend(rgb_path)
            tiff_paths.extend(tiff_path)
            nmap_paths.extend(nmap_path)
            
        rgb_paths.sort()
        tiff_paths.sort()
        nmap_paths.sort()
        
        sample_paths = list(zip(rgb_paths, tiff_paths, nmap_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label, cls = self.img_paths[idx], self.labels[idx], self.cls_list[idx]
        rgb_path = img_path[0]
        # tiff_path = img_path[1]
        nmap_path = img_path[2]
        text_prompt = "A photo of a " + cls
        
        #load image data
        img = Image.open(rgb_path).convert('RGB')
        nmap = Image.open(nmap_path).convert('RGB')
        img, nmap = self.paired_transform(img, nmap)
        
        return img, nmap, text_prompt

class MVTec3DTest(Dataset):
    def __init__(self, img_size, dataset_path, class_name):
        self.img_size = img_size
        self.class_name = class_name
        self.img_path = os.path.join(dataset_path, self.class_name, 'test')
        self.rgb_transform = transforms.Compose(
        [transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()])
        self.gt_transform = transforms.Compose( 
        [transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                nmap_paths = glob.glob(os.path.join(self.img_path, defect_type, 'nmap') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                nmap_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths, nmap_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                nmap_paths = glob.glob(os.path.join(self.img_path, defect_type, 'nmap') + "/*.png")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                nmap_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths, nmap_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))
                
        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        nmap_path = img_path[2]
        text_prompt = f"A photo of a {self.class_name}"
        
        #load image data
        img_original = Image.open(rgb_path).convert('RGB')
        nmap_original = Image.open(nmap_path).convert('RGB')
        img = self.rgb_transform(img_original)
        nmap = self.rgb_transform(nmap_original)
        
        if gt == 0:
            gt = torch.zeros([1, self.img_size, self.img_size])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)
        return (img, nmap, text_prompt), gt[:1], label

def mvtec3D_train_loader(args):
    dataset = MVTec3DTrain(args.image_size, args.data_path, split='train', is_memory=False)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, drop_last=True, pin_memory=True)
    return data_loader

def mvtec3D_val_loader(args):
    dataset = MVTec3DTrain(args.image_size, args.data_path, split='validation', is_memory=False)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, drop_last=True, pin_memory=True)
    return data_loader

def mvtec3D_test_loader(args, class_name, split):
    if split == "memory":
        dataset = MVTec3DTrain(img_size=args.image_size, dataset_path=args.data_path, class_name=class_name, split='train', is_memory=True)
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, drop_last=False, pin_memory=True)
    elif split == "test":
        dataset = MVTec3DTest(args.image_size, args.data_path, class_name=class_name)
        data_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=args.workers, shuffle=False, drop_last=False,pin_memory=True)
    return data_loader
