import open3d as o3d
from utils.mvtec3d_util import *


# 读取PLY点云文件
orginized_pc = read_tiff_organized_pc("/mnt/home_6T/public/jayliu0313/datasets/mvtec3d_preprocessing/bagel/train/good/xyz/000.tiff")
resized_organized_pc = resize_organized_pc(orginized_pc)
unorganized_pc_no_zeros , nonzero_idx = orgpc_to_unorgpc(resized_organized_pc)
o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
# 读取RGB纹理图像
texture = o3d.io.read_image("/mnt/home_6T/public/jayliu0313/datasets/mvtec3d_preprocessing/bagel/train/good/rgb/000.png")

# 创建一个几何对象来组合点云和纹理信息
geometry = o3d.geometry.TriangleMesh()
geometry.vertices = o3d_pc.points
geometry.textures = [o3d.geometry.Image(texture)]

# 创建一个可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window()

# 添加几何对象到可视化窗口
vis.add_geometry(geometry)

# 设置渲染参数
render_options = vis.get_render_option()
# render_options.load_from_json("your_render_option_json_file.json")  # 可选，设置渲染参数

# 运行渲染循环
vis.run()

# 关闭可视化窗口
vis.destroy_window()