from core.data import test_lightings_loader
from core.reconstruct_method import Nmap_Repair, Nmap_Rec, RGB_Nmap_Rec, Vae_Rec
from core.diffuision_method import ControlNet_Rec, Diffusion_Rec, DDIMInv_Rec
from core.memory_method import Memory_Method
from core.ddim_memory_method import DDIM_Memory
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
        if args.method_name == "rgb_nmap_rec":
            self.method = RGB_Nmap_Rec(args, cls_path)
        elif args.method_name == "nmap_repair":
            self.method = Nmap_Repair(args, cls_path)
        elif args.method_name == "nmap_rec":
            self.method = Nmap_Rec(args, cls_path)
        elif args.method_name == "memory":
            self.method = Memory_Method(args, cls_path)
        elif args.method_name == "ddim_memory":
            self.method = DDIM_Memory(args, cls_path)
        elif args.method_name == "vae_rec":
            self.method = Vae_Rec(args, cls_path)
        elif args.method_name == "controlnet_rec":
            self.method = ControlNet_Rec(args, cls_path)
        elif args.method_name == "diffusion_rec":
            self.method = Diffusion_Rec(args, cls_path)
        elif args.method_name == "ddiminv_rec":
            self.method = DDIMInv_Rec(args, cls_path)
        # elif args.method_name == "mean_rec":
        #     self.method = Mean_Rec(args, cls_path)
        # elif args.method_name == "rec":
        #     self.method = Rec(args, cls_path)
        else:
            return TypeError
        self.cls = cls
        self.log_file = open(osp.join(cls_path, "class_score.txt"), "a", 1)
        self.method_name = args.method_name
    
    def fit(self):
        dataloader = test_lightings_loader(self.args, self.cls, "memory")
        with torch.no_grad():
            for i, (lightings, _, text_prompt) in enumerate(tqdm(dataloader, desc=f'Extracting train features for class {self.cls}')):
                # if i == 5:
                #     break
                self.method.add_sample_to_mem_bank(lightings, text_prompt)
      
        self.method.run_coreset()

    def evaluate(self):
        dataloader = test_lightings_loader(self.args, self.cls, "test")
        
        for i, ((images, nmap, text_prompt), gt, label) in enumerate(tqdm(dataloader)):
            # if i == 5:
            #     break
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
