import tifffile as tiff
import torch

def reweight(img_size, nonzero_indices, weight = 0.0):
    reweight_arr = torch.full((3, img_size * img_size), weight)
    reweight_arr[:, nonzero_indices] = 1
    reweight_arr = reweight_arr.reshape(3, img_size, img_size)
    return reweight_arr

def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])


def read_tiff_organized_pc(path):
    tiff_img = tiff.imread(path)
    return tiff_img

def orgpc_to_unorgpc(organized_pc):
    organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    
    pc_no_zeros = unorganized_pc[nonzero_indices, :]
    return pc_no_zeros, nonzero_indices

def get_zero_indices(organized_pc):
    organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
    zero_indices = np.nonzero(np.all(unorganized_pc == 0, axis=1))[0]
    return zero_indices

def resize_organized_pc(organized_pc, target_height=224, target_width=224, tensor_out=True):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0).contiguous()
    torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=(target_height, target_width),
                                                                 mode='bicubic')
    if tensor_out:
        return torch_resized_organized_pc.squeeze(dim=0).contiguous()
    else:
        return torch_resized_organized_pc.squeeze().permute(1, 2, 0).contiguous().numpy()


def organized_pc_to_depth_map(organized_pc):
    return organized_pc[:, :, 2]

import imageio.v3 as iio
import numpy as np
from sklearn import preprocessing

def load_and_convert_normals(normal_img, pose_txt):
    # input pose
    pose = np.loadtxt(pose_txt)

    # input normals
    normals = iio.imread(normal_img).astype(float)
    img_shape = normals.shape

    # [0, 255] -> [-1, 1] and normalize
    normals = preprocessing.normalize(normals / 127.5 - 1.0, norm="l2")

    # flatten, flip Z and Y, then apply the pose
    normals = normals.reshape(-1, 3) @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    normals = normals @ pose[:3, :3].T

    # back to image, if needed
    normals = normals.reshape(img_shape)
    return normals
