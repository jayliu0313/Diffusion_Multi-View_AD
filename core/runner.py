from core.data import test_lightings_loader
from core.reconstruct import Mean_Reconstruct, Reconstruct
import torch
import os
import os.path as osp

class Runner():
    def __init__(self, args, model, cls):
        cls_path = os.path.join(args.output_dir, cls)
        if not os.path.exists(cls_path):
            os.makedirs(cls_path)

        self.args = args
        if args.method_name == "mean_reconstruct":
            self.method = Mean_Reconstruct(args, model, cls_path)
        else:
            self.method = Reconstruct(args, model, cls_path)
        self.cls = cls
        self.log_file = open(osp.join(cls_path, "class_score.txt"), "a", 1)
        self.method_name = args.method_name
        
    def evaluate(self):
        dataloader = test_lightings_loader(self.args, self.cls)
        with torch.no_grad():
            for i, ((images, pc), gt, label) in enumerate(dataloader):
                self.method.predict(i, images, gt, label)

        image_rocauc, pixel_rocauc, au_pro = self.method.calculate_metrics()
        self.method.visualizae_heatmap()
        image_rocaucs = dict()
        pixel_rocaucs = dict()
        au_pros = dict()
        image_rocaucs[self.method_name] = round(image_rocauc, 3)
        pixel_rocaucs[self.method_name] = round(pixel_rocauc, 3)
        au_pros[self.method_name] = round(au_pro, 3)

        self.log_file.write(
            f'Class: {self.cls} {self.method_name}, Image ROCAUC: {image_rocauc:.3f}, Pixel ROCAUC: {pixel_rocauc:.3f}, AUPRO:  {au_pro:.3f}'
        )
        self.log_file.close()
        return image_rocaucs, pixel_rocaucs, au_pros
