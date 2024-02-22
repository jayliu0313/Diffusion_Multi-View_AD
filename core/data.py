import os
import glob
from PIL import Image
from torchvision import transforms
from einops import repeat, rearrange
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from utils.utils import CutPaste
from torch.utils.data import DataLoader
import numpy as np
import random

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]

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

# not use
class MemoryLightings(BaseDataset):
    def __init__(self, class_name, img_size, dataset_path, is_alignment=False):
        super().__init__(split="train", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.data_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cutpaste = CutPaste(type="custom")
        self.is_alignment = is_alignment
        
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
        if self.is_alignment and idx % 2 == 0:
            images = self.cutpaste(images)
            n_map = self.cutpaste(n_map)
            
        aug_imgs = [self.rgb_transform(img) for img in images]
        aug_nmap = [self.rgb_transform(nmap) for nmap in n_map]
        aug_imgs = torch.stack(aug_imgs)
        aug_nmap = torch.cat(aug_nmap)
        return aug_imgs, aug_nmap, text_prompt

# load training image   
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
        # images = images*2.0 - 1.0
        
        normal_path = img_path[1]
        depth_path = img_path[2]
        normal = Image.open(normal_path).convert('RGB')
        nmap = self.rgb_transform(normal)
        return images, nmap, text_prompt

# load validation image using for training 
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

# load training normal map
class TrainNmap(Dataset):
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
        normal_paths = []

        for cls in eyecandies_classes():
            normal_paths.extend(glob.glob(os.path.join(self.img_path, cls, 'train', 'data') + "/*_normals.png"))
        
        normal_paths.sort()     
        
        sample_paths = list(zip(normal_paths))
        
        data_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return data_tot_paths, tot_labels

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path, label = self.data_paths[idx], self.labels[idx]
        nmap_path = img_path[0]
        nmap = Image.open(nmap_path).convert('RGB')
        nmap = self.rgb_transform(nmap)
        return nmap

# load training normal map using for training 
class ValNmap(Dataset):
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
        normal_paths = []

        for cls in eyecandies_classes():
            normal_paths.extend(glob.glob(os.path.join(self.img_path, cls, 'val', 'data') + "/*_normals.png"))

        normal_paths.sort()     

        sample_paths = list(zip(normal_paths))
    
        data_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return data_tot_paths, tot_labels

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path, label = self.data_paths[idx], self.labels[idx]
        nmap_path = img_path[0]
        nmap = Image.open(nmap_path).convert('RGB')
        nmap = self.rgb_transform(nmap)
        return nmap

def test_lightings_loader(args, cls, split):
    if split == 'memory':
        dataset = MemoryLightings(cls, args.image_size, args.data_path, False)
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False,
                                pin_memory=True)
    elif split == 'memory_align':
        dataset = MemoryLightings(cls, args.image_size, args.data_path, False)
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=args.workers, drop_last=False,
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

def train_nmap_loader(args):
    dataset = TrainNmap(args.image_size, args.data_path)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, drop_last=True,
                              pin_memory=True)
    return data_loader

def val_nmap_loader(args):
    dataset = ValNmap(args.image_size, args.data_path)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, drop_last=True,
                              pin_memory=True)
    return data_loader


######################################################
#                     MVTEC 3D-AD                    #
######################################################
class MVTec3DTrain(Dataset):
    def __init__(self, img_size, dataset_path, cls=None, split='train'):
        self.img_size = img_size
        self.img_path = dataset_path
        self.split = split
        self.cls = cls
        self.rgb_transform = transforms.Compose(
        [transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        ])
        self.cls_list = []
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []

        rgb_paths = []
        tiff_paths = []

        if self.cls == None:
            cls_list = mvtec3d_classes()
        else:
            cls_list = [self.cls]
        
        for cls in cls_list:
            rgb_path = glob.glob(os.path.join(self.img_path, cls, self.split, 'good', 'rgb') + "/*.png")
            tiff_path = glob.glob(os.path.join(self.img_path, cls, self.split, 'good', 'xyz') + "/*.tiff")
            self.cls_list.extend([cls] * len(rgb_path))
            rgb_paths.extend(rgb_path)
            tiff_paths.extend(tiff_path)
            
        rgb_paths.sort()
        tiff_paths.sort()
        
        sample_paths = list(zip(rgb_paths, tiff_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label, cls = self.img_paths[idx], self.labels[idx], self.cls_list[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        text_prompt = "A photo of a " + cls
        #load image data
        img = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img)
        return img, torch.zeros_like(img), text_prompt

class MVTec3DTest(Dataset):
    def __init__(self, img_size, dataset_path, cls):
        self.img_size = img_size
        self.cls = cls
        self.img_path = os.path.join(dataset_path, self.cls, 'test')
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
                rgb_paths.sort()
                tiff_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))
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
        text_prompt = f"A photo of a {self.cls}"
        #load image data
        img_original = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img_original)

        if gt == 0:
            gt = torch.zeros([1, self.img_size, self.img_size])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)
        return (img, torch.zeros_like(img), text_prompt), gt[:1], label

def mvtec3D_train_loader(args):
    dataset = MVTec3DTrain(args.image_size, args.data_path, split='train')
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, drop_last=True, pin_memory=True)
    return data_loader

def mvtec3D_val_loader(args):
    dataset = MVTec3DTrain(args.image_size, args.data_path, split='validation')
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, drop_last=True, pin_memory=True)
    return data_loader

def mvtec3D_test_loader(args, cls, split):
    if split == "memory":
        dataset = MVTec3DTrain(args.image_size, args.data_path, cls=cls, split='train')
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, drop_last=False,pin_memory=True)
    elif split == "test":
        dataset = MVTec3DTest(args.image_size, args.data_path, cls=cls)
        data_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=args.workers, shuffle=False, drop_last=False,pin_memory=True)
    return data_loader

######################################################
#                      MVTEC AD                      #
######################################################
MVTEC_AD_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

class MVTecDataset(Dataset):
    def __init__(
        self,
        source,
        classname,
        imagesize=256,
        split="train",
        is_val=False
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.is_val = is_val
        self.classnames_to_use = [classname] if classname is not None else MVTEC_AD_CLASSNAMES
        self.train_val_split = 0.8

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.rgb_transform = transforms.Compose(
        [transforms.Resize((imagesize, imagesize), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()])
        self.gt_transform = transforms.Compose( 
        [transforms.Resize((imagesize, imagesize), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()])
        
        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.rgb_transform(image)
        text_prompt = f"A photo of a {classname}"

        if self.split == "test" and mask_path is not None:
            mask = Image.open(mask_path)
            mask = self.gt_transform(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])
            
        if self.split == "test":
            return (image, torch.zeros_like(image), text_prompt), mask, int(anomaly != "good")
        else:
            return image, torch.zeros_like(image), text_prompt
        
    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == "train" and self.is_val == False:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == "train" and self.is_val == True:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == "test" and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == "test" and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
    
def mvtec_train_loader(args):
    dataset = MVTecDataset(source=args.data_path, classname=None, imagesize=args.image_size, split='train')
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, drop_last=True, pin_memory=True)
    return data_loader

def mvtec_val_loader(args):
    dataset = MVTecDataset(source=args.data_path, classname=None, imagesize=args.image_size, split='train', is_val=True)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, drop_last=True, pin_memory=True)
    return data_loader

def mvtec_test_loader(args, cls, split):
    if split == "memory":
        dataset = MVTecDataset(source=args.data_path, classname=cls, imagesize=args.image_size, split='train')
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, drop_last=False, pin_memory=True)
    elif split == "test":
        dataset = MVTecDataset(source=args.data_path, classname=cls, imagesize=args.image_size, split='test')
        data_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=args.workers, shuffle=False, drop_last=False, pin_memory=True)
    return data_loader