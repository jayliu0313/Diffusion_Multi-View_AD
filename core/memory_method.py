import torch
from tqdm import tqdm
from core.base import Base_Method
from utils.utils import t2np
from sklearn import random_projection
from utils.utils import KNNGaussianBlur
from core.models.backnone import RGB_Extractor
from core.models.network_util import Decom_Block

class Memory_Method(Base_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.decomp_block = Decom_Block(768).to(self.device)
        self.decomp_block.requires_grad_(False)
        self.decomp_block.eval()
        
        # Create Feature Extractor
        self.feature_extractor = RGB_Extractor(device=self.device, backbone_name=args.backbone_name)
        checkpoint_dict = torch.load(args.load_decomp_ckpt, map_location=self.device)
        if checkpoint_dict['backbone'] is not None:
            print("load backbone checkpoints!")
            self.feature_extractor.load_state_dict(checkpoint_dict['backbone'])
            
        if  checkpoint_dict['decomp_block'] is not None:
            print("load decomp checkpoints!")
            self.decomp_block.load_state_dict(checkpoint_dict['decomp_block'])
        
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval() 
        self.feature_extractor.requires_grad_(False)
        self.f_coreset = 1
        self.coreset_eps = 0.9
        self.n_reweight = 3
        self.blur = KNNGaussianBlur(4)
        
    def compute_s_s_map(self, patch, feature_map_dims):
        # patch = (patch - self.mean)/self.std
        # self.patch_lib = self.rgb_layernorm(self.patch_lib)
        # print(patch.shape)
        # print(self.patch_lib.shape)
        # patch = patch.to('cpu')
        self.patch_lib = self.patch_lib.to(self.device)
        dist = torch.cdist(patch, self.patch_lib)
 
        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = w * s_star
        
        # segmentation map
        
        
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')
        s_map = self.blur(s_map.to('cpu'))
        return s.to('cpu'), s_map

    def get_memory_nnfeature(self, patch, feature_map_dims):
        dist = torch.cdist(patch, self.patch_lib.to(self.device))
        min_idx = torch.argmin(dist, dim=1)
        nnfeature = self.patch_lib[min_idx].view(1, 4, *feature_map_dims)
        nnfeature = nnfeature.repeat(6, 1, 1, 1)
        # print(nnfeature.shape)
        return nnfeature.to(self.device)
    
    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0)
        # self.mean = torch.mean(self.patch_lib)
        # self.std = torch.std(self.patch_lib)
        # self.patch_lib = (self.patch_lib - self.mean)/self.std

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
        img_input = self.image_transform(lightings)
        latents = self.feature_extractor(img_input)
        rgb_feature_maps = self.decomp_block.get_meanfc(latents, keep_dim=False)
        # rgb_feature_maps = self.average(rgb_feature_maps)
        rgb_patch = rgb_feature_maps.reshape(rgb_feature_maps.shape[1], -1).T
        # print(rgb_patch.shape)
        self.patch_lib.append(rgb_patch.to('cpu'))
    
    def predict(self, i, lightings, nmap, gt, label):
        # img = lightings[:, 5, :, :, :].squeeze(0)
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        img_input = self.image_transform(lightings)
        latents = self.feature_extractor(img_input)
        rgb_feature_maps = self.decomp_block.get_meanfc(latents, keep_dim=False)
        rgb_patch = rgb_feature_maps.reshape(rgb_feature_maps.shape[1], -1).T
        s, smap = self.compute_s_s_map(rgb_patch, rgb_feature_maps.shape[-2:])
        img = lightings[5, :, :, :]
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(smap))
        self.pixel_labels.extend(t2np(gt)) 