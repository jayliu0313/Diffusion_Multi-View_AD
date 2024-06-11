import torch
import torch.nn as nn
from torchvision import transforms
from core.base import DDIM_Method
from utils.utils import t2np, nxn_cos_sim
from core.models.controllora import  ControlLoRAModel
from torch.optim.adam import Adam
# from scipy.stats import wasserstein_distance
# from scipy.spatial.distance import cdist
from utils.ptp_utils import *
from geomloss import SamplesLoss
from utils.visualize_util import display_one_img
# from sinkhorn import sinkhorn
# # from pyemd import emd
# from dist_matrix.cuda_dist_matrix_full import dist_matrix as gpu_dist_matrix
# import ot
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import ot

import os
import os.path as osp
np.seterr(divide='ignore', invalid='ignore')

class Memory_Method(DDIM_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.f_coreset = 1
        self.coreset_eps = 0.9
        self.n_reweight = 3
        self.target_timestep = max(args.noise_intensity)
        # print(self.target_timestep)
        self.patch_lib = []
        self.nmap_patch_lib = []
        
        
    def compute_s_s_map(self, patch, patch_lib, feature_map_dims, p=2, k=1):
        _, _, C, _, _ = patch.shape
        target_patch_lib = patch_lib[-1].permute(1, 0, 2, 3).reshape(C, -1).T.to(self.device)
        target_patch = patch[-1].permute(1, 0, 2, 3).reshape(C, -1).T.to(self.device)
       
        # target_patch_lib = patch_lib[-1].to(self.device)
        # target_patch = patch[-1].to(self.device)
        dist = torch.cdist(target_patch, target_patch_lib, p=p)
        smap, min_idx = torch.topk(dist, k=k, largest=False)
        smap = smap[:, -1]
        min_idx = min_idx[:, -1]
        min_idx = min_idx.to('cpu')
        if len(patch_lib) > 1:
            mul_smap = self.pdist(patch.to(self.device), patch_lib[:, min_idx, :].to(self.device))
            smap, _ = torch.max(mul_smap, dim=0)
        #s_star = torch.max(smap)
        topk_value, _ = torch.topk(smap, k=self.topk)
        s_star = torch.mean(topk_value)
        if self.reweight:
            s_idx = torch.argmax(smap)
            m_test = target_patch[s_idx].unsqueeze(0)  # anomalous patch
            m_star = target_patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, target_patch_lib)  # find knn to m_star pt.1
            _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

            m_star_knn = torch.linalg.norm(m_test - target_patch_lib[nn_idx[0, 1:]], dim=1)
            D = torch.sqrt(torch.tensor(target_patch.shape[1]))
            w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
            s_star = w * s_star
        smap = smap.view(1, 1, *feature_map_dims)
        smap = torch.nn.functional.interpolate(smap, size=(self.image_size, self.image_size), mode='bilinear')
        smap = self.blur(smap.to('cpu'))

        return s_star.to('cpu'), smap

    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib , 1)
        if self.nmap_patch_lib:
            self.nmap_patch_lib = torch.cat(self.nmap_patch_lib , 1)

    @torch.no_grad()
    def ddim_loop(self, latents, text_emb):
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        latent_list = []
        t_list = []
        for t in reversed(self.timesteps_list):
            noise_pred = self.unet(latents.to(self.device), t, text_emb)['sample']
            latents = self.next_step(noise_pred, t, latents)
            if t in self.mul_timesteps:
                latent_list.append(latents.to('cpu'))
                t_list.append(t)
        return latent_list, t_list
    
    @torch.no_grad()
    def get_feature_layers(self, latent, t, text_emb):
        pred_f = self.unet(latent.to(self.device), t, text_emb)['up_ft']
        resized_f = []
        for i in range(len(pred_f)):
            if i in self.feature_layers:
                largest_fmap_size = pred_f[self.feature_layers[-1]].shape[-2:]
                resized_f.append(torch.nn.functional.interpolate(pred_f[i], largest_fmap_size, mode='bicubic').to('cpu'))
        features = torch.cat(resized_f, dim=1)
        return features
        
    @torch.no_grad() 
    def get_unet_f(self, latents, text_emb, islighting=True):
        latent_list, t_list = self.ddim_loop(latents, text_emb)
        unetf_list = []
        for i, t in enumerate(t_list):

            unet_f = self.get_feature_layers(latent_list[i], t, text_emb)
            B, C, H, W = unet_f.shape
            if islighting and self.data_type=="eyecandies":
                unet_f = torch.mean(unet_f.view(-1, 6, C, H, W), dim=1)
            # unet_f = unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
            unetf_list.append(unet_f.to('cpu'))

        unet_fs = torch.stack(unetf_list)
        return unet_fs



class DDIM_Memory(Memory_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        
    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        latents = self.image2latents(lightings)
        bsz = latents.shape[0]
        text_embeddings = self.get_text_embedding(text_prompt, bsz)
        _, timesteps, noisy_latents = self.forward_process_with_T(latents, self.timesteps_list[-1])
        # print(timesteps)
        model_output = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings,
        )
        
        unet_f = model_output['up_ft'][3]
        B, C, H, W = unet_f.shape
        
        train_unet_f = torch.mean(unet_f.view(-1, 6, C, H, W), dim=1)
        train_unet_f = train_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T
        self.patch_lib.append(train_unet_f.cpu())
    
    def predict(self, i, lightings, nmap, text_prompt, gt, label):
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
    
        latents = self.image2latents(lightings)
        cond_embeddings = self.get_text_embedding(text_prompt, 6)

    
        _, timesteps, noisy_latents = self.forward_process_with_T(latents, self.timesteps_list[-1])
        # print(timesteps)
        model_output = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=cond_embeddings,
        )
        
        unet_f = model_output['up_ft'][3]
        B, C, H, W = unet_f.shape
        
        test_unet_f = torch.mean(unet_f.view(-1, 6, C, H, W), dim=1)
        test_unet_f = test_unet_f.permute(1, 0, 2, 3).reshape(C, -1).T

        s, smap = self.compute_s_s_map(test_unet_f, self.patch_lib, unet_f.shape[-2:])
        img = lightings[5, :, :, :]
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(smap))
        self.pixel_labels.extend(t2np(gt))

class DDIMInvRGB_Memory(Memory_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)

    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):
        lightings = lightings.to(self.device)
        #lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        latents = self.image2latents(lightings)
        bsz = latents.shape[0]
        text_emb = self.get_text_embedding(text_prompt, bsz)
        
        unet_f = self.get_unet_f(latents, text_emb, islighting=False)
        self.patch_lib.append(unet_f.cpu())
    
    def predict(self, i, lightings, nmap, text_prompt, gt, label):
        lightings = lightings.to(self.device)
        #lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        
        latents = self.image2latents(lightings)
        text_emb = self.get_text_embedding(text_prompt, 1)
        unet_f = self.get_unet_f(latents, text_emb, islighting=False)
        
        s, smap = self.compute_s_s_map(unet_f, self.patch_lib, latents.shape[-2:])
        img = lightings[0]
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(smap))
        self.pixel_labels.extend(t2np(gt))

class DDIMInvLoco_Memory(Memory_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.g_pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.g_patch_lib = []
        self.pos_patch_lib = []
        self.g_feature_layers = args.g_feature_layers
        self.num_class = args.num_class
        # self.topk = args.topk
        self.criteria = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        self.kmeans = KMeans(
           n_clusters=self.num_class,
           n_init='auto'
        )
        self.pca = PCA(n_components=2)
        self.cls_path = cls_path

    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib , 1)
        self.pos_patch_lib = torch.cat(self.pos_patch_lib, 0)
        self.g_patch_lib = torch.cat(self.g_patch_lib , 0)

    @torch.no_grad()
    def get_feature_layers(self, latent, t, text_emb):
        pred_f = self.unet(latent.to(self.device), t, text_emb)['up_ft']
        resized_f = []
        g_resized_f = []
        largest_fmap_size = pred_f[self.feature_layers[-1]].shape[-2:]
        g_largest_fmap_size = pred_f[self.g_feature_layers[-1]].shape[-2:]
        # print(largest_fmap_size)
        for i in range(len(pred_f)):
            if i in self.feature_layers:
                resized_f.append(torch.nn.functional.interpolate(pred_f[i], largest_fmap_size, mode='bicubic').to('cpu'))
            if i in self.g_feature_layers:
                g_resized_f.append(torch.nn.functional.interpolate(pred_f[i], g_largest_fmap_size, mode='bicubic').to('cpu'))

        g_features = torch.cat(g_resized_f, dim=1)
        features = torch.cat(resized_f, dim=1)

        return features, g_features
        
    @torch.no_grad() 
    def get_unet_f(self, latents, text_emb):
        latent_list, t_list = self.ddim_loop(latents, text_emb)
        unetf_list = []
        g_unetf_list = []
        for i, t in enumerate(t_list):
            unet_f, g_unet_f = self.get_feature_layers(latent_list[i], t, text_emb)
            unetf_list.append(unet_f.to('cpu'))
            g_unetf_list.append(g_unet_f.to('cpu'))
        unet_fs = torch.stack(unetf_list)
        g_unetf_fs= torch.stack(g_unetf_list)
        return unet_fs, g_unetf_fs

    def compute_global_pdist(self, g_patch, g_patch_lib, patch, patch_lib, feature_map_dims, p=2, k=20):
        patch = patch[-1]
        patch_lib = patch_lib[-1]
        B, C, H, W = g_patch.shape
        g_target_patch_lib = g_patch_lib.permute(1, 0, 2, 3).reshape(C, -1).T.to(self.device)
        g_target_patch = g_patch.permute(1, 0, 2, 3).reshape(C, -1).T.to(self.device)
        dist = torch.cdist(g_target_patch, g_target_patch_lib, p=p)
        _, min_idx = torch.topk(dist, k=k, largest=False)
        B, C, H, W = patch.shape
        patch_lib = patch_lib.to(self.device)
        patch = patch.to(self.device)
        nn_feature_map = patch_lib[min_idx[0, :]]

        patch = patch.permute(0, 2, 3, 1)
        nn_feature_map = nn_feature_map.permute(0, 2, 3, 1)
        multi_smap = self.pdist(patch, nn_feature_map)

        smap, _ = torch.min(multi_smap, dim=0)

        s, _ = torch.max(smap.flatten(), dim=0)
        smap = smap.view(1, 1, *feature_map_dims)
        smap = torch.nn.functional.interpolate(smap, size=(self.image_size, self.image_size), mode='bilinear')
        smap = self.blur(smap.to('cpu'))

        return s.to('cpu'), smap

    def compute_global_cdist(self, g_patch, g_patch_lib, patch, patch_lib, feature_map_dims, p=2, k=20):
        # Global Patch
        B, C, H, W = g_patch.shape
        g_target_patch_lib = g_patch_lib.permute(1, 0, 2, 3).reshape(C, -1).T.to(self.device)
        g_target_patch = g_patch.permute(1, 0, 2, 3).reshape(C, -1).T.to(self.device)
        dist = torch.cdist(g_target_patch, g_target_patch_lib, p=p)
        _, min_idx = torch.topk(dist, k=k, largest=False)

        # Local Patch
        patch = patch[-1]
        patch_lib = patch_lib[-1]
        B, C, H, W = patch.shape
        patch_lib = patch_lib.to(self.device)
        patch = patch.to(self.device)
        nn_feature_map = patch_lib[min_idx[0, :]]
        
        patch = patch.permute(0, 2, 3, 1).reshape(H*W, C)
        nn_feature_map = nn_feature_map.permute(0, 2, 3, 1).reshape(k, H*W, C)

        
        # .reshape(C, -1).T
        # dist = torch.cdist(patch, nn_feature_map)
        smap = torch.zeros((k, H*W))

        for j in range(k):
            target_nnfmap = nn_feature_map[j]
            used_flags = torch.zeros(H*W)
            for i in range(H*W):
                dist = torch.cdist(patch[i:i+1], target_nnfmap[used_flags == 0])
                min_dist, min_idx = torch.min(dist.flatten(), dim=0)
                used_flags[min_idx] = 1
                smap[j, i] = min_dist

        smap, _ = torch.min(smap, dim=0)
        # print(smap.max())
        s, _ = torch.max(smap, dim=0)
        smap = smap.view(1, 1, *feature_map_dims)
        smap = torch.nn.functional.interpolate(smap, size=(self.image_size, self.image_size), mode='bilinear')
        smap = self.blur(smap.to('cpu'))

        return s.to('cpu'), smap
    
    def compute_emd(self, g_patch, g_patch_lib, patch, patch_lib, image, item, feature_map_dims, p=2, k=20):
        # Global Patch
        B, C, H, W = g_patch.shape
        g_target_patch_lib = g_patch_lib.permute(1, 0, 2, 3).reshape(-1, C).to(self.device)
        g_target_patch = g_patch.permute(1, 0, 2, 3).reshape(-1, C).to(self.device)
        dist = torch.cdist(g_target_patch, g_target_patch_lib, p=p)
        _, min_idx = torch.topk(dist, k=k, largest=False)

        # Local Patch
        patch_lib = patch_lib[-1]
        patch = patch[-1]
        B, C, H, W = patch.shape

        # Local Training patch
        nn_feature_map = patch_lib[min_idx[0, :]]
        train_labels = self.train_labels[min_idx[0, :]]
        labels, counts = np.unique(train_labels, return_counts=True)
        train_hist = np.zeros(self.num_class, dtype=int)
        train_hist[labels] = counts

        nn_patch_feature = nn_feature_map.permute(0, 2, 3, 1).reshape(k, H*W, C)
        nn_patch_feature = nn_patch_feature[0]

        # Local testing patch
        patch = patch.permute(0, 2, 3, 1).reshape(H*W, C)
        test_prd_labels = self.kmeans.predict(patch)
        labels, counts = np.unique(test_prd_labels, return_counts=True)
        # 創建一個與原始類別數量一致的零填充的結果陣列
        test_hist = np.zeros(self.num_class, dtype=int)

        # 將實際統計結果填入對應的位置
        test_hist[labels] = counts
        
 
        # visualize the label of the image
        # if self.viz and item % 20 == 0:
        #     test_prd_labels_tensor= torch.tensor(test_prd_labels)
        #     test_prd_labels_tensor = test_prd_labels_tensor.view(1, 1, *feature_map_dims)
        #     self.transform_mask = transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.NEAREST)
        #     test_prd_labels_tensor = self.transform_mask(test_prd_labels_tensor)
        #     image = t2np(image)
        #     test_prd_labels_img = t2np(test_prd_labels_tensor)
        #     for class_index in range(self.num_class):
        #         class_index_arr = np.full_like(test_prd_labels_img, class_index)
        #         class_mask = (test_prd_labels_img == class_index_arr)
        #         class_mask = np.repeat(class_mask, 3, axis=1)

        #         masked_image = image * class_mask
        #         display_one_img(image[0], masked_image[0], os.path.join(self.cls_path, "Reconstruction", str(item) + "_" + str(class_index) +".png"))

        patch_cls_mean_f = torch.zeros((self.num_class, C), dtype=torch.float32)

        # test coordinates
        test_coordinates = np.indices((H, W))
        test_coordinates = np.expand_dims(test_coordinates, axis=0)
        test_coordinates = torch.tensor(test_coordinates).permute(0, 2, 3, 1).reshape(-1, 2)

        test_cls_pos_hist = np.zeros((self.num_class, H, W), dtype=np.float32)  # 初始化 class_counts
        for class_index in range(self.num_class):
            # find mean test feature of each cluster
            indices = np.where(test_prd_labels == class_index)  # 找到屬於該類別的索引
            class_features = patch[indices[0]]   # 獲取屬於該類別的特徵
            mean_features = torch.mean(class_features, dim=0)  # 計算該類別的平均特徵
            patch_cls_mean_f[class_index] = mean_features

            # count position
            # 獲取屬於當前類別的位置座標
            class_positions = test_coordinates[indices[0]]
            # 將位置座標在對應的類別統計數組中進行累加
            np.add.at(test_cls_pos_hist, (class_index, class_positions[:, 0], class_positions[:, 1]), 1)
    
        train_cls_pos_hist = self.trian_cls_pos_hist.reshape(self.num_class, -1)
        test_cls_pos_hist = test_cls_pos_hist.reshape(self.num_class, -1)

        train_class_total = np.sum(train_cls_pos_hist, axis=1, keepdims=True)
        test_class_total = np.sum(test_cls_pos_hist, axis=1, keepdims=True)
        train_class_total = np.repeat(train_class_total, H*W, axis=1) 
        test_class_total = np.repeat(test_class_total, H*W, axis=1) 


        train_cls_pos_hist = np.divide(train_cls_pos_hist, train_class_total, out=np.zeros_like(train_cls_pos_hist), where=(train_class_total != 0))
        test_cls_pos_hist = np.divide(test_cls_pos_hist, test_class_total, out=np.zeros_like(test_cls_pos_hist), where=(test_class_total != 0))

        # train_cls_pos_hist = torch.tensor(train_cls_pos_hist).to(self.device)
        # test_cls_pos_hist = torch.tensor(test_cls_pos_hist).to(self.device)

        # pos_cost = torch.cdist(train_cls_pos_hist, test_cls_pos_hist)
        # pos_cost = t2np(pos_cost)

        # center_tensor = torch.tensor(self.centers, dtype=torch.float32).to(self.device)
        # dist_cost = torch.cdist(patch_cls_mean_f.to(self.device), center_tensor)
        # dist_cost = t2np(dist_cost)

        dist = torch.cdist(patch.to(self.device), nn_patch_feature.to(self.device))
        dist = t2np(dist)
        pos_cost = np.zeros((len(test_cls_pos_hist), len(train_cls_pos_hist)), dtype=np.float32)

        for i in range(len(test_cls_pos_hist)):
            cur_test = test_cls_pos_hist[i]
            for j in range(len(train_cls_pos_hist)):
                cur_train = train_cls_pos_hist[j]
                if(np.sum(cur_test) == 0.0):
                    random_values = np.random.rand(1024)
                    cur_test = random_values  / np.sum(random_values)
                
                if(np.sum(cur_train) == 0.0):
                    random_values = np.random.rand(1024)
                    cur_train = random_values  / np.sum(random_values)

                _, log = ot.emd(cur_test, cur_train, M=dist, log=True)
                pos_cost[i][j] = log['cost']
            
        total_cost = pos_cost
 
        test_hist = test_hist.astype(np.float32)
        train_hist = train_hist.astype(np.float32)
    
        _, log = ot.emd(test_hist, train_hist, M=total_cost, log=True)
        s = log['cost']

        # emd_dist = torch.tensor(emd_dist)

        smap = torch.zeros((H, W))
        smap = smap.view(1, 1, *feature_map_dims)
        smap = torch.nn.functional.interpolate(smap, size=(self.image_size, self.image_size), mode='bilinear')\
        
        s = torch.tensor(np.array(s))
        # s = torch.max(emd_dist)[0]
        return s, smap

    def add_sample_to_mem_bank(self, images, nmap, text_prompt):
        images = images.to(self.device)

        latents = self.image2latents(images)
        bsz = latents.shape[0]
        text_emb = self.get_text_embedding(text_prompt, bsz)
        
        unet_f, g_unet_f = self.get_unet_f(latents, text_emb)
        g_unet_f = self.g_pool(g_unet_f[-1]).cpu()

        unet_f = unet_f.cpu()
        _, B, _, H, W = unet_f.shape
        coordinates = np.indices((H, W))
        coordinates = np.expand_dims(coordinates, axis=0)
        coordinates = np.repeat(coordinates, B, axis=0)
        coordinates = torch.from_numpy(coordinates)

        self.pos_patch_lib.append(coordinates)
        self.g_patch_lib.append(g_unet_f)
        self.patch_lib.append(unet_f)
    
    def predict(self, i, images, nmap, text_prompt, gt, label):
        images = images.to(self.device)
        
        latents = self.image2latents(images)
        text_emb = self.get_text_embedding(text_prompt, 1) 
        unet_f, g_unet_f = self.get_unet_f(latents, text_emb)
        g_unet_f = self.g_pool(g_unet_f[-1])
        # s, smap  = self.compute_s_s_map(unet_f, self.patch_lib, latents.shape[-2:])
        # print(smap.shape)
        g_s, gmap = self.compute_emd(g_unet_f, self.g_patch_lib, unet_f, self.patch_lib, images, i, latents.shape[-2:],k=self.topk)
        # print(g_smap.shape)
        # final_s = s * g_s
        # #final_s = g_s * s

        final_s = g_s
        final_smap = gmap
        img = images[0]
        
        self.image_labels.append(label.numpy())
        self.image_preds.append(final_s.numpy())
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(final_smap))
        self.pixel_labels.extend(t2np(gt))

    def cluster_training_data(self):
        _, B, C, H ,W = self.patch_lib.shape
        # print(self.patch_lib.shape)
        patch_lib = self.patch_lib.permute(0, 1, 3, 4, 2).reshape(-1, C)
        pos_lib = self.pos_patch_lib.permute(0, 2, 3, 1).reshape(-1, 2)

        train_labels = self.kmeans.fit_predict(patch_lib)

        # print(train_labels.shape)
        self.train_labels = train_labels.reshape(B, H, W)
        
        self.centers = self.kmeans.cluster_centers_

        # pca_data = self.pca.fit_transform(patch_lib)
        # plt.figure(figsize=(16, 12))

        self.trian_cls_pos_hist = np.zeros((self.num_class, H, W), dtype=np.float32)  # 初始化 class_counts
        
        for i in range(self.num_class):
            # 找到屬於當前類別的位置索引
            class_indices = np.where(train_labels == i)
            # 獲取屬於當前類別的位置座標
            class_positions = pos_lib[class_indices[0]]
            # 將位置座標在對應的類別統計數組中進行累加
            np.add.at(self.trian_cls_pos_hist, (i, class_positions[:, 0], class_positions[:, 1]), 1)
            # plt.scatter(pca_data[train_labels == i, 0], pca_data[train_labels == i, 1], label=f'Cluster {i}', s=0.1)

        # plt.title('PCA Visualization of KMeans Clusters')
        # plt.xlabel('PCA Component 1')
        # plt.ylabel('PCA Component 2')
        # plt.legend()
        # plt.savefig(self.cls_path)

        # print(self.patch_lib.shape)
 

class DDIMInvNmap_Memory(Memory_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)

    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):
        nmap = nmap.to(self.device)
        latents = self.image2latents(nmap)
        bsz = latents.shape[0]
        text_emb = self.get_text_embedding(text_prompt, bsz)
        
        unet_f = self.get_unet_f(latents, text_emb, islighting=False)
        self.patch_lib.append(unet_f.cpu())


    def predict(self, i, lightings, nmap, text_prompt, gt, label):
        nmap = nmap.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        latents = self.image2latents(nmap)
        text_emb = self.get_text_embedding(text_prompt, 1)
        unet_f = self.get_unet_f(latents, text_emb, islighting=False)
        s, smap = self.compute_s_s_map(unet_f, self.patch_lib, latents.shape[-2:])
        img = lightings[5, :, :, :]
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.image_list.append(t2np(img))
        self.pixel_preds.append(t2np(smap))
        self.pixel_labels.extend(t2np(gt))
        # print(img)
        
class DDIMInvUnified_Memory(Memory_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)
        self.weight = 1
        self.bias = 0
    
    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):
        text_emb = self.get_text_embedding(text_prompt, 1)
        
        # rgb
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_unet_f(rgb_latents, rgb_text_embs, islighting=True)
        self.patch_lib.append(rgb_unet_f.cpu())
        
        # normal  map
        nmap = nmap.to(self.device)
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_unet_f(nmap_latents, nmap_text_embs, islighting=False)
        self.nmap_patch_lib.append(nmap_unet_f.cpu())

    def predict_align_data(self, lightings, nmap, text_prompt):
        text_emb = self.get_text_embedding(text_prompt, 1)
        # rgb
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_unet_f(rgb_latents, rgb_text_embs)
        rgb_s, rgb_smap = self.compute_s_s_map(rgb_unet_f, self.patch_lib, rgb_latents.shape[-2:], k=2)
        
        # nromal map
        nmap = nmap.to(self.device)
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_unet_f(nmap_latents, nmap_text_embs)
        nmap_s, nmap_smap = self.compute_s_s_map(nmap_unet_f, self.nmap_patch_lib, nmap_latents.shape[-2:], k=2)

        # image_level
        self.nmap_image_preds.append(nmap_s.numpy())
        self.rgb_image_preds.append(rgb_s.numpy())
        # pixel_level
        self.rgb_pixel_preds.extend(rgb_smap.flatten().numpy())
        self.nmap_pixel_preds.extend(nmap_smap.flatten().numpy())

    @ torch.no_grad()
    def predict(self, i, lightings, nmap, text_prompt, gt, label):
        text_emb = self.get_text_embedding(text_prompt, 1)
        # rgb
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_unet_f(rgb_latents, rgb_text_embs, islighting=True)
        rgb_s, rgb_smap = self.compute_s_s_map(rgb_unet_f, self.patch_lib, rgb_latents.shape[-2:])
        
        # normal map
        nmap = nmap.to(self.device)
        nmap_latents = self.image2latents(nmap)
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_unet_f(nmap_latents, nmap_text_embs, islighting=False)
        nmap_s, nmap_smap = self.compute_s_s_map(nmap_unet_f, self.nmap_patch_lib, nmap_latents.shape[-2:])

        pixel_map = rgb_smap + nmap_smap
        s = rgb_s + nmap_s
        
        ### Record Score ###
        img = lightings[-1, :, :, :]
        self.image_list.append(t2np(img))
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.rgb_image_preds.append(rgb_s.numpy())
        self.nmap_image_preds.append(nmap_s.numpy())
        
        self.pixel_labels.append(t2np(gt))
        self.pixel_preds.append(t2np(pixel_map))
        self.rgb_pixel_preds.append(t2np(rgb_smap))
        self.nmap_pixel_preds.append(t2np(nmap_smap))

class ControlNet_DDIMInv_Memory(Memory_Method):
    def __init__(self, args, cls_path):
        super().__init__(args, cls_path)

        # Setting ControlNet Model 
        print("Loading ControlNet")
        self.controllora = ControlLoRAModel.from_unet(self.unet, lora_linear_rank=args.controllora_linear_rank, lora_conv2d_rank=args.controllora_conv2d_rank)
        self.controllora.load_state_dict(torch.load(args.load_controlnet_ckpt, map_location=self.device))
        self.controllora.tie_weights(self.unet)
        self.controllora.requires_grad_(False)
        self.controllora.eval()
        self.controllora.to(self.device)

        self.weight = args.rgb_weight
        self.bias = 0
        self.nmap_weight = args.nmap_weight
        
    @torch.no_grad()
    def get_feature_layers(self, latent, condition_map, t, text_emb):
        pred_f = self.controlnet(latent.to(self.device), condition_map, t, text_emb)['up_ft']
        features_list = []
        for layer in self.feature_layers:
            resized_f = torch.nn.functional.interpolate(pred_f[layer], size=(32, 32), mode='bicubic')
            features_list.append(resized_f.to('cpu'))
        features = torch.cat(features_list, dim=1)
        return features
    
    @torch.no_grad()    
    def controlnet(self, noisy_latents, condition_map, timestep, text_emb):
        down_block_res_samples, mid_block_res_sample = self.controllora(
            noisy_latents, timestep,
            encoder_hidden_states=text_emb,
            controlnet_cond=condition_map,
            guess_mode=False, return_dict=False,
        )
        model_output = self.unet(
            noisy_latents, timestep,
            encoder_hidden_states=text_emb,
            down_block_additional_residuals=[sample for sample in down_block_res_samples],
            mid_block_additional_residual=mid_block_res_sample
        )
        return model_output
    
    @torch.no_grad()
    def controlnet_ddim_loop(self, latents, condition_map, text_emb):
        latents = latents.clone().detach()
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        latent_list = []
        t_list = []
        for t in reversed(self.timesteps_list):
            noise_pred = self.controlnet(latents.to(self.device), condition_map, t, text_emb)['sample']
            latents = self.next_step(noise_pred, t, latents)
            if t in self.mul_timesteps:
                latent_list.append(latents.to('cpu'))
                t_list.append(t)
        return latent_list, t_list
    
    @torch.no_grad() 
    def get_controlnet_f(self, latents, condition_map, text_emb, islighting=True):
        noisy_latents, t_list  = self.controlnet_ddim_loop(latents, condition_map, text_emb)
        control_fs = []
        for i, t in enumerate(t_list):
            contol_f = self.get_feature_layers(noisy_latents[i].to(self.device), condition_map, t, text_emb)
            B, C, H, W = contol_f.shape
            if islighting and self.data_type=="eyecandies":
                contol_f = torch.mean(contol_f.view(-1, 6, C, H, W), dim=1)
            control_fs.append(contol_f.to('cpu'))
        control_fs = torch.stack(control_fs)
        return control_fs
    
    def add_sample_to_mem_bank(self, lightings, nmap, text_prompt):

        text_emb = self.get_text_embedding(text_prompt, 1)
        
        lightings = lightings.to(self.device)
        
        nmap = nmap.to(self.device)

        # single_lightings = lightings[:, 5, :, :, :] # [bs, 3, 256, 256]
        if self.data_type == "eyecandies":
            nmap_repeat = nmap.repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]
        else:
            nmap_repeat = nmap
        # rgb
        lightings = lightings.view(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_controlnet_f(rgb_latents, nmap_repeat, rgb_text_embs)
        self.patch_lib.append(rgb_unet_f.cpu())

        # normal  map
        nmap_latents = self.image2latents(nmap)
        if self.data_type == "eyecandies":
            nmap_latents = nmap_latents.repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]      
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_controlnet_f(nmap_latents, lightings, nmap_text_embs)
        self.nmap_patch_lib.append(nmap_unet_f.cpu())  

    def predict(self, i, lightings, nmap, text_prompt, gt, label):
        
        text_emb = self.get_text_embedding(text_prompt, 1)
        lightings = lightings.to(self.device)
        nmap = nmap.to(self.device)

        # single_lightings = lightings[:, 5, :, :, :] # [bs, 3, 256, 256]
        if self.data_type == "eyecandies":
            nmap_repeat = nmap.repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]
        else:
            nmap_repeat = nmap
        # rgb
        lightings = lightings.to(self.device)
        lightings = lightings.reshape(-1, 3, self.image_size, self.image_size)
        rgb_latents = self.image2latents(lightings)
        bsz = rgb_latents.shape[0]
        rgb_text_embs = text_emb.repeat(bsz, 1, 1)
        rgb_unet_f = self.get_controlnet_f(rgb_latents, nmap_repeat, rgb_text_embs)
        rgb_s, rgb_smap = self.compute_s_s_map(rgb_unet_f, self.patch_lib, rgb_latents.shape[-2:])
        
        # normal map
        nmap = nmap.to(self.device)
        nmap_latents = self.image2latents(nmap)
        if self.data_type == "eyecandies":
            nmap_latents = nmap_latents.repeat_interleave(6, dim=0) # [bs * 6, 3, 256, 256]      
        bsz = nmap_latents.shape[0]
        nmap_text_embs = text_emb.repeat(bsz, 1, 1)
        nmap_unet_f = self.get_controlnet_f(nmap_latents, lightings, nmap_text_embs)
        nmap_s, nmap_smap = self.compute_s_s_map(nmap_unet_f, self.nmap_patch_lib, nmap_latents.shape[-2:])

        ### Combine RGB and Nmap score map ###
        s = rgb_s * self.weight + nmap_s * self.nmap_weight
        smap = rgb_smap + nmap_smap
        # s, _ = torch.topk(smap, k=self.topk)
        # s = torch.mean(s)
        img = lightings[-1, :, :, :]
        self.image_list.append(t2np(img))
        self.image_labels.append(label.numpy())
        self.image_preds.append(s.numpy())
        self.rgb_image_preds.append(rgb_s.numpy())
        self.nmap_image_preds.append(nmap_s.numpy())
        
        self.pixel_labels.append(t2np(gt))
        self.pixel_preds.append(t2np(smap))
        self.rgb_pixel_preds.append(t2np(rgb_smap))
        self.nmap_pixel_preds.append(t2np(nmap_smap))
