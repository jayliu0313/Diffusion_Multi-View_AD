from core.data import test_lightings_loader
from core.ddim_recconstruct_method import *
from core.ddim_memory_method import *
from tqdm import tqdm
import torch
import os
import os.path as osp

class Runner():
    def __init__(self, args, cls):
        cls_path = os.path.join(args.output_dir, cls)
        if not os.path.exists(cls_path):
            os.makedirs(cls_path)

        self.args = args
        
        if args.method_name == "ddim_memory":
            self.method = DDIM_Memory(args, cls_path)
        elif args.method_name == "ddiminvrgb_memory":
            self.method = DDIMInvRGB_Memory(args, cls_path)
        elif args.method_name == "ddiminvnmap_memory":
            self.method = DDIMInvNmap_Memory(args, cls_path)
        elif args.method_name == "ddiminvrgbnmap_memory":
            self.method = DDIMInvRGBNmap_Memory(args, cls_path)
        elif args.method_name == "ddiminvunified_memory":
            self.method = DDIMInvUnified_Memory(args, cls_path)
        elif args.method_name == "controlnet_ddiminv_memory":
            self.method = ControlNet_DDIMInv_Memory(args, cls_path)
        elif args.method_name == "controlnet_rec":
            self.method = ControlNet_Rec(args, cls_path)
        elif args.method_name == "ddim_rec":
            self.method = DDIM_Rec(args, cls_path)
        elif args.method_name == "nullinv_rec":
            self.method = NULLInv_Rec(args, cls_path)
        elif args.method_name == "controlnet_directinv_memory":
            self.method = ControlNet_DirectInv_Memory(args, cls_path)
        elif args.method_name == "directinv_memory":
            self.method = DirectInv_Memory(args, cls_path)
        else:
            return TypeError
        
        self.cls = cls
        self.log_file = open(osp.join(cls_path, "class_score.txt"), "a", 1)
        self.method_name = args.method_name
    
    def fit(self):
        dataloader = test_lightings_loader(self.args, self.cls, "memory", self.args.batch_size, False)
        with torch.no_grad():
            for i, (lightings, nmap, text_prompt) in enumerate(tqdm(dataloader, desc=f'Extracting train features for class {self.cls}')):
                # if i == 5:
                #     break
                text_prompt = f'A photo of a {self.cls}'
                # text_prompt = ""
                self.method.add_sample_to_mem_bank(lightings, nmap, text_prompt)
            self.method.run_coreset()

    def alignment(self):
        dataloader = test_lightings_loader(self.args, self.cls, "memory", 1, True)
        with torch.no_grad():
            print(f'Computing weight and bias for alignment')
            for i, (lightings, nmap, text_prompt) in enumerate(tqdm(dataloader, desc=f'Extracting train features for class {self.cls}')):
                if i == 25:
                    break  
                text_prompt = f'A photo of a {self.cls}'  
                self.method.predict_align_data(lightings, nmap, text_prompt)
            self.method.cal_alignment()
        
    def evaluate(self):
        dataloader = test_lightings_loader(self.args, self.cls, "test", 1, False)
        
        for i, ((images, nmap, text_prompt), gt, label) in enumerate(tqdm(dataloader)):
            # if i == 5:
            #     break
            text_prompt = f'A photo of a {self.cls}'
            # text_prompt = ""
            self.method.predict(i, images, nmap, text_prompt, gt, label)

        image_rocauc, pixel_rocauc, au_pro = self.method.calculate_metrics()
        total_rec_loss = self.method.get_rec_loss()
        rec_mean_loss = total_rec_loss / len(dataloader)

        self.method.visualizae_heatmap()

        image_rocaucs = dict()
        pixel_rocaucs = dict()
        au_pros = dict()
        rec_losses = dict()
        image_rocaucs[self.method_name] = round(image_rocauc, 3)
        pixel_rocaucs[self.method_name] = round(pixel_rocauc, 3)
        au_pros[self.method_name] = round(au_pro, 3)
        rec_losses[self.method_name] = round(rec_mean_loss, 6)

        self.log_file.write(
            f'Class: {self.cls} {self.method_name}, Image ROCAUC: {image_rocauc:.3f}, Pixel ROCAUC: {pixel_rocauc:.3f}, AUPRO:  {au_pro:.3f}\n'
            f'Reconstruction Loss: {rec_mean_loss}'
        )
        self.log_file.close()
        return image_rocaucs, pixel_rocaucs, au_pros, rec_losses
