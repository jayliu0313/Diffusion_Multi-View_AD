import open3d as o3d

import cv2
import os
import numpy as np
from PIL import Image 
from utils.mvtec3d_util import getPointCloud
from tqdm import tqdm
DATASET_PATH = ""
RESULT_PATH = ""


CLASS = [
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



def load_pontsRGB(self, points_path, image_path):
    unorg_pc, nonzero_idx = getPointCloud(points_path)
    rgb_image  = Image.open(image_path).convert('RGB')
    

    img = np.array(rgb_image).reshape(-1,3)[nonzero_idx] / 255.0
    pcd = o3d.geometry.PointCloud()
    
    points = np.asarray(unorg_pc).reshape(-1,3)
    colors = np.asarray(img).reshape(-1,3)

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    #R = pcd.get_rotation_matrix_from_xyz((0, np.pi, np.pi))
    #pcd.rotate(R, center=(0, 0, 0))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01,max_nn=50))

    image_np = np.array(rgb_image)
    out_norm_o3d = np.zeros_like(image_np.reshape(-1, 3), dtype=np.float32)
    out_norm_o3d[nonzero_idx] = np.array(pcd.normals).reshape(-1, 3)
    normal_map = out_norm_o3d.reshape(image_np.shape)

    error_normal_pos = normal_map[:, :, 2] < 0
    normal_map[error_normal_pos] = normal_map[error_normal_pos] * -1.0

    normal_map = normal_map + 1
    normal_map = normal_map / 2
    normal_map = (normal_map * 255).astype(np.uint8)
    #cv2.imwrite('surface_normal_o3d.png', normal_map)
    cv2.imshow('surface_normal_o3d.png', normal_map)
    cv2.waitKey(0)
    return pcd

def trans_to_nmap(data_path, save_path):

    file_name = os.path.basename(data_path).split('.')[0]
    rgb_path = data_path.replace("xyz", "rgb").replace("tiff", "png")

    # Load RGB Image and Point Cloud
    rgb_image  = np.array(Image.open(rgb_path).convert('RGB'))
    unorg_pc, nonzero_idx = getPointCloud(data_path)

    # Apply point cloud format using open3d
    pcd = o3d.geometry.PointCloud()
    points = np.asarray(unorg_pc).reshape(-1,3)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01,max_nn=50))

    # Integrate normal vector into a normal map
    out_norm_o3d = np.zeros_like(rgb_image.reshape(-1, 3), dtype=np.float32) + 2.0
    out_norm_o3d[nonzero_idx] = np.array(pcd.normals).reshape(-1, 3)
    normal_map = out_norm_o3d.reshape(rgb_image.shape)

    # Fix incorrect normal vector
    error_normal_pos = normal_map[:, :, 2] < 0
    normal_map[error_normal_pos] = normal_map[error_normal_pos] * -1.0

    # Normalize and Save
    normal_map = np.where(normal_map == 2.0, -1.0, normal_map)
    normal_map = normal_map + 1
    normal_map = normal_map / 2
    normal_map = (normal_map * 255).astype(np.uint8)
    
    cv2.imwrite(os.path.join(save_path, file_name + "_normal.png"), normal_map)
       
import glob
        
if __name__ == "__main__":

    for cls_name in CLASS:  
        # Training Data #
        tiff_paths = glob.glob(os.path.join(DATASET_PATH, cls_name, "train", 'good', 'xyz') + "/*.tiff")
        save_dir = os.path.join(RESULT_PATH, cls_name, "train", 'good', 'nmap')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for tiff_file in tqdm(tiff_paths, desc=f"{cls_name} Training Data"):
            trans_to_nmap(data_path=tiff_file, save_path=save_dir)

        # Validation Data #
        tiff_paths = glob.glob(os.path.join(DATASET_PATH, cls_name, "validation", 'good', 'xyz') + "/*.tiff")
        save_dir = os.path.join(RESULT_PATH, cls_name, "validation", 'good', 'nmap')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for tiff_file in tqdm(tiff_paths, desc=f"{cls_name} validation Data"):
            trans_to_nmap(data_path=tiff_file, save_path=save_dir)

        # Testing Data #
        test_cls_path = os.path.join(DATASET_PATH, cls_name, "test")
        defect_types = os.listdir(test_cls_path)
        for defect_type in defect_types:

            tiff_paths = glob.glob(os.path.join(test_cls_path, defect_type, 'xyz') + "/*.tiff")

            save_dir = os.path.join(RESULT_PATH, cls_name, "test", defect_type, 'nmap')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for tiff_file in tqdm(tiff_paths, f"{cls_name} Testing Data type:{defect_type}"):
                trans_to_nmap(data_path=tiff_file, save_path=save_dir)