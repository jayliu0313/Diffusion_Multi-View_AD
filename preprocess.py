import os
import cv2
import numpy as np
import tifffile
import yaml
import imageio.v3 as iio
import math
import argparse
import glob

# The same camera has been used for all the images
FOCAL_LENGTH = 711.11

def load_and_convert_depth(depth_img, info_depth):
    with open(info_depth) as f:
        data = yaml.safe_load(f)
    mind, maxd = data["normalization"]["min"], data["normalization"]["max"]

    dimg = iio.imread(depth_img)
    dimg = dimg.astype(np.float32)
    dimg = dimg / 65535.0 * (maxd - mind) + mind
    return dimg

def depth_to_pointcloud(depth_img, info_depth, pose_txt, focal_length):

    # input depth map (in meters) --- cfr previous section
    depth_mt = load_and_convert_depth(depth_img, info_depth)

    # input pose
    pose = np.loadtxt(pose_txt)

    # camera intrinsics
    height, width = depth_mt.shape[:2]
    intrinsics_4x4 = np.array([
        [focal_length, 0, width / 2, 0],
        [0, focal_length, height / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]
    )

    # build the camera projection matrix
    camera_proj = intrinsics_4x4 @ pose

    # build the (u, v, 1, 1/depth) vectors (non optimized version)
    camera_vectors = np.zeros((width * height, 4))
    count=0
    for j in range(height):
        for i in range(width):
            camera_vectors[count, :] = np.array([i, j, 1, 1/depth_mt[j, i]])
            count += 1

    # invert and apply to each 4-vector
    hom_3d_pts= np.linalg.inv(camera_proj) @ camera_vectors.T
    # print(hom_3d_pts.shape)
    # remove the homogeneous coordinate
    pcd = depth_mt.reshape(-1, 1) * hom_3d_pts.T
    return pcd[:, :3]

def depth_to_pcnv(depth_img, color_img, nv_img, info_depth, pose_txt, focal_length):
    # input depth map (in meters) --- cfr previous section
    depth_mt = load_and_convert_depth(depth_img, info_depth)

    # input pose
    pose = np.loadtxt(pose_txt)

    # camera intrinsics
    height, width = depth_mt.shape[:2]
    intrinsics_4x4 = np.array([
        [focal_length, 0, width / 2, 0],
        [0, focal_length, height / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]
    )

    # build the camera projection matrix
    camera_proj = intrinsics_4x4 @ pose

    # build the (u, v, 1, 1/depth) vectors (non optimized version)
    camera_vectors = np.zeros((width * height, 4))
    count=0
    for j in range(height):
        for i in range(width):
            camera_vectors[count, :] = np.array([i, j, 1, 1/depth_mt[j, i]])
            count += 1

    # invert and apply to each 4-vector
    hom_3d_pts= np.linalg.inv(camera_proj) @ camera_vectors.T
    # print(hom_3d_pts.shape)
    # remove the homogeneous coordinate
    pcd = depth_mt.reshape(-1, 1) * hom_3d_pts.T
    return pcd[:, :3]

def remove_point_cloud_background(pc):

    # The second dim is z
    dz =  pc[256,1] - pc[-256,1]
    dy =  pc[256,2] - pc[-256,2]

    norm =  math.sqrt(dz**2 + dy**2)
    start_points = np.array([0, pc[-256, 1], pc[-256, 2]])
    cos_theta = dy / norm
    sin_theta = dz / norm

    # Transform and rotation
    rotation_matrix = np.array([[1, 0, 0], [0, cos_theta, -sin_theta],[0, sin_theta, cos_theta]])
    processed_pc = (rotation_matrix @ (pc - start_points).T).T

    # Remove background point
    for i in range(processed_pc.shape[0]):
        if processed_pc[i,1] > -0.02:
            processed_pc[i, :] = -start_points
        if processed_pc[i,2] > 1.8:
            processed_pc[i, :] = -start_points
        elif processed_pc[i,0] > 1 or processed_pc[i,0] < -1:
            processed_pc[i, :] = -start_points

    processed_pc = (rotation_matrix.T @ processed_pc.T).T + start_points

    index = [0, 2, 1]
    processed_pc = processed_pc[:,index]
    return processed_pc*[0.1, -0.1, 0.1]

def process_data(category_dir):
    depth_paths = glob.glob(category_dir + "/*depth.png")
    info_depth_paths = glob.glob(category_dir + "/*info_depth.yaml")
    pose_paths = glob.glob(category_dir + "/*pose.txt")
    depth_paths.sort()
    info_depth_paths.sort()
    pose_paths.sort()
    for i in range(len(depth_paths)):
        pc = depth_to_pointcloud(
                depth_paths[i],
                info_depth_paths[i],
                pose_paths[i],
                FOCAL_LENGTH,
            )
        pc = remove_point_cloud_background(pc)
        pc = pc.reshape(512, 512, 3)
        tifffile.imwrite(os.path.join(category_dir, str(i).zfill(3)+'.tiff'), pc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_path', default='/mnt/home_6T/public/jayliu0313/datasets/Eyecandies/', type=str, help="Eyecandies dataset path.")

    args = parser.parse_args()
    
    categories_list = ['GummyBear', 'ChocolateCookie']
    for categorie in categories_list:
        train_cat_dir = os.path.join(args.dataset_path, categorie, "train", "data")
        test_pub_cat_dir = os.path.join(args.dataset_path, categorie, "test_public", "data")
        test_priv_cat_dir = os.path.join(args.dataset_path, categorie, "test_private", "data")

        # training data process
        process_data(train_cat_dir)

        # # testing public data process
        process_data(test_pub_cat_dir)

        # testing private data process
        process_data(test_priv_cat_dir)

        

        