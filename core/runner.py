from core.data import test_lightings_loader, mvtec3D_test_loader, mvtec_test_loader, mvtecLoco_test_loader
from core.ddim_recconstruct_method import *
from core.ddim_memory_method import *
from tqdm import tqdm
import torch
import os
import os.path as osp

class Runner():
    def __init__(self, args, cls, modality_names):
        cls_path = os.path.join(args.output_dir, cls)
        if not os.path.exists(cls_path):
            os.makedirs(cls_path)

        self.args = args
        self.modality_names = modality_names
        
        if args.method_name == "ddim_memory":
            self.method = DDIM_Memory(args, cls_path)
        elif args.method_name == "ddiminvrgb_memory":
            self.method = DDIMInvRGB_Memory(args, cls_path)
        elif args.method_name == "ddiminvnmap_memory":
            self.method = DDIMInvNmap_Memory(args, cls_path)
        elif args.method_name == "ddiminvunified_memory":
            self.method = DDIMInvUnified_Memory(args, cls_path)
        elif args.method_name == "controlnet_ddiminv_memory":
            self.method = ControlNet_DDIMInv_Memory(args, cls_path)
        else:
            raise TypeError
        
        self.cls = cls

        self.log_file = open(osp.join(cls_path, "class_score.txt"), "a", 1)
        self.method_name = args.method_name
        
        if args.dataset_type == "eyecandies":
            self.memory_loader = test_lightings_loader(args, cls, "memory")
            self.test_loader = test_lightings_loader(args, cls, "test")
        elif args.dataset_type == "mvtec3d":
            self.memory_loader = mvtec3D_test_loader(args, cls, "memory")
            self.test_loader = mvtec3D_test_loader(args, cls, "test")
        
        if args.is_align:
            args.batch_size = 1
            self.align_loader = mvtec3D_test_loader(args, cls, "memory") 
            
    def fit(self):
        with torch.no_grad():
            for i, (lightings, nmap, text_prompt) in enumerate(tqdm(self.memory_loader, desc=f'Extracting train features for class {self.cls}')):
                # if i == 4:
                #     break
                text_prompt = f'A photo of a {self.cls}'
                self.method.add_sample_to_mem_bank(lightings, nmap, text_prompt)
            self.method.run_coreset()
            self.method.cluster_training_data()

    def alignment(self):
        with torch.no_grad():
            print(f'Computing weight and bias for alignment')
            for i, (lightings, nmap, text_prompt) in enumerate(tqdm(self.align_loader, desc=f'Extracting train features for class {self.cls}')):
                if i == 25:
                    break
                text_prompt = f'A photo of a {self.cls}'  
                self.method.predict_align_data(lightings, nmap, text_prompt)
            self.method.cal_alignment()
        
    def evaluate(self):
        
        with torch.no_grad():
            for i, ((images, nmap, text_prompt), gt, label) in enumerate(tqdm(self.test_loader, desc="Extracting test features")):
                # if i == 105:
                #     break
                text_prompt = f'A photo of a {self.cls}'
                self.method.predict(i, images, nmap, text_prompt, gt, label)

        if self.args.viz:
            for modality_name in self.modality_names:
                self.method.visualizae_heatmap(modality_name, self.cls)
            
        image_rocaucs = dict()
        pixel_rocaucs = dict()
        au_pros = dict()
        rec_losses = dict()
        for modality_name in self.modality_names:
            image_rocauc, pixel_rocauc, au_pro = self.method.calculate_metrics(modality_name, self.cls)
            image_rocaucs[modality_name] = round(image_rocauc, 3)
            pixel_rocaucs[modality_name] = round(pixel_rocauc, 3)
            au_pros[modality_name] = round(au_pro, 3)
            self.log_file.write(f'Class: {self.cls} {modality_name}, Image ROCAUC: {image_rocauc:.3f}, Pixel ROCAUC: {pixel_rocauc:.3f}, AUPRO:  {au_pro:.3f}\n')
        self.log_file.close()
        return image_rocaucs, pixel_rocaucs, au_pros, rec_losses
