import torch
import tqdm
from core.base import Base_Method
from utils.utils import t2np
from sklearn import random_projection
from utils.utils import KNNGaussianBlur

class Memory_Method(Base_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.f_coreset = 1
        

    def compute_s_s_map(self, patch, feature_map_dims):
        dist = torch.cdist(patch, self.patch_lib.to(self.device))
        dist = dist.to('cpu')
        min_val, min_idx = torch.min(dist, dim=1)
        s_star = torch.max(min_val)
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')
        s_map = self.blur(s_map)
        return s_star, s_map

    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0)
        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_lib = self.patch_lib[self.coreset_idx]

    def get_coreset_idx_randomp(self, z_lib, n=1000, eps=0.90, float16=True, force_cpu=False):
        """Returns n coreset idx for given z_lib.
        Performance on AMD3700, 32GB RAM, RTX3080 (10GB):
        CPU: 40-60 it/s, GPU: 500+ it/s (float32), 1500+ it/s (float16)
        Args:
            z_lib:      (n, d) tensor of patches.
            n:          Number of patches to select.
            eps:        Agression of the sparse random projection.
            float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
            force_cpu:  Force cpu, useful in case of GPU OOM.
        Returns:
            coreset indices
        """

        print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
        try:
            transformer = random_projection.SparseRandomProjection(eps=eps)
            z_lib = torch.tensor(transformer.fit_transform(z_lib))
            print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
        except ValueError:
            print("   Error: could not project vectors. Please increase `eps`.")

        select_idx = 0
        last_item = z_lib[select_idx:select_idx + 1]
        coreset_idx = [torch.tensor(select_idx)]
        min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
        # The line below is not faster than linalg.norm, although i'm keeping it in for
        # future reference.
        # min_distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True)

        if float16:
            last_item = last_item.half()
            z_lib = z_lib.half()
            min_distances = min_distances.half()
        if torch.cuda.is_available() and not force_cpu:
            last_item = last_item.to("cuda")
            z_lib = z_lib.to("cuda")
            min_distances = min_distances.to("cuda")

        for _ in tqdm(range(n - 1)):
            distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)  # broadcasting step
            min_distances = torch.minimum(distances, min_distances)  # iterative step
            select_idx = torch.argmax(min_distances)  # selection step

            # bookkeeping
            last_item = z_lib[select_idx:select_idx + 1]
            min_distances[select_idx] = 0
            coreset_idx.append(select_idx.to("cpu"))
        return torch.stack(coreset_idx)
   
    def add_sample_to_mem_bank(self, lightings):
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        six_fc = self.model.get_fc(lightings)
        rgb_feature_maps = self.model.get_mean_fc(six_fc)
        rgb_patch = self.average(rgb_feature_maps)

        rgb_patch_size28 = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        self.patch_lib.append(rgb_patch_size28.to('cpu'))
    
    def predict(self, item, lightings, gt, label):
        img = lightings[:, 5, :, :, :].squeeze(0)
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        six_fc = self.model.get_fc(lightings)
        rgb_feature_maps = self.model.get_mean_fc(six_fc)
        rgb_patch = self.average(rgb_feature_maps)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        s, smap = self.compute_s_s_map(rgb_patch, rgb_feature_maps[0].shape[-2:])
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(smap))
        self.pixel_labels.extend(t2np(gt))