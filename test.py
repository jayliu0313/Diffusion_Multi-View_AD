import os
import torch
import datetime
import argparse
import pandas as pd
import os.path as osp
from core.runner import Runner
from utils.utils import set_seeds, log_args
os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser(description='test')

DEBUG = False

# Dataset and environment setup
parser.add_argument('--data_path', default="/mnt/home_6T/public/jayliu0313/datasets/Eyecandies/", type=str)
# /mnt/home_6T/public/jayliu0313/datasets/Eyecandies/
parser.add_argument('--output_dir', default="./output")
parser.add_argument('--dataset_type', default="mvtec3d", help="eyecandies, mvtec3d, mvtec2d")
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument("--workers", default=4)
parser.add_argument('--CUDA', type=int, default=0, help="choose the device of CUDA")
parser.add_argument('--viz', action="store_true")
parser.add_argument('--seed', type=int, default=7)

# Method choose
parser.add_argument('--method_name', default="ddiminvrgb_memory", help=" \
Reconstruction Base: controlnet_rec, ddim_rec, nullinv_rec, \
DDIM Base: ddim_memory, ddiminvrgb_memory, ddiminvnmap_memory, \
ddiminvunified_memory, ddiminvunified_timefusion_memory, controlnet_ddiminv_memory\
")
parser.add_argument('--reweight', default=False, type=bool)
parser.add_argument('--score_type', default=0, type=int, help="0 is max score, 1 is mean score") # just for score map, max score: maximum each pixel of 6 score maps, mean score: mean of 6 score maps 
parser.add_argument('--feature_layers', default=[3], type=int)
parser.add_argument('--topk', default=1, type=int)

#### Load Checkpoint ####
parser.add_argument("--load_vae_ckpt", default="")
parser.add_argument("--load_unet_ckpt", default="checkpoints/diffusion_checkpoints/TrainMVTec3DAD_UnetV1-5_Aug_RandomApply/best_unet.pth")
# "/mnt/home_6T/public/jayliu0313/check_point/Diffusion_ckpt/TrainUNet_ClsText_FeatureLossAllLayer_AllCls_epoch130/best_unet.pth"
# "/mnt/home_6T/public/jayliu0313/check_point/Diffusion_ckpt/TrainUnifiedUNet_ClsText_FeatureLossAllLayer_allcls/best_unet.pth"
# "/mnt/home_6T/public/jayliu0313/check_point/Diffusion_ckpt/TrainUnifiedUNet_ClsText_ckpt6_allcls/best_unet.pth"
# "checkpoints/diffusion_checkpoints/TrainMVTec2D_UnetV1-5_woAug/best_unet.pth"
#  "/home/samchu0218/Multi_Lightings/checkpoints/unet_model/MVTec/epoch_unet.pth"

parser.add_argument('--load_controlnet_ckpt', type=str, default="/home/samchu0218/Multi_Lightings/checkpoints/controlnet_model/with_woFlossUnet/epoch51c_ontrolnet.pth")
# "/home/samchu0218/Multi_Lightings/checkpoints/controlnet_model/with_woFlossUnet/epoch51_controlnet.pth"
# "/home/samchu0218/Multi_Lightings/checkpoints/controlnet_model/with_woFLossUnet_woFLoss/epoch36_controlnet.pth"
# Unet Model (Diffusion Model)
parser.add_argument("--diffusion_id", type=str, default="runwayml/stable-diffusion-v1-5", help="CompVis/stable-diffusion-v1-4, runwayml/stable-diffusion-v1-5")

parser.add_argument("--revision", type=str, default="", help="v1-4:ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c, v1-5:null")

# parser.add_argument("--noise_intensity", type=int, default=81)
parser.add_argument("--noise_intensity", type=int, default=[61, 81])
parser.add_argument("--step_size", type=int, default=20)

# Controlnet Model Setup
parser.add_argument("--controllora_linear_rank", type=int, default=4)
parser.add_argument("--controllora_conv2d_rank", type=int, default=0)

# test_t = [[1], [21], [41], [61], [1, 21]]
# test_t = [[81, 101, 121], [81, 101]]
test_t = [[21], [41], [81]]
def run(args):
    if args.dataset_type=='eyecandies':
        classes = [
        # 'CandyCane',
        # 'ChocolateCookie',
        'ChocolatePraline',
        # 'Confetto',
        # 'GummyBear',
        # 'HazelnutTruffle',
        # 'LicoriceSandwich',
        # 'Lollipop',
        # 'Marshmallow',
        # 'PeppermintCandy'
        ]
    elif args.dataset_type=='mvtec3d':
        classes = [
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
        args.data_path = "/mnt/home_6T/public/jayliu0313/datasets/mvtec3d_preprocessing/"
    elif args.dataset_type=='mvtec2d':
        classes = [
            "bottle",
            "cable",
            "capsule",
            "carpet",
            "grid",
            "hazelnut",
            "leather",
            "metal_nut",
            "pill",
            "screw",
            "tile",
            "toothbrush",
            "transistor",
            "wood",
            "zipper",
        ]
        args.data_path = "/mnt/home_6T/public/samchu0218/Raw_Datasets/MVTec_AD/MVTec_2D/"
    result_file = open(osp.join(args.output_dir, "results.txt"), "a", 1)   
    MODALITY_NAMES = ['RGB', 'Nmap', 'RGB+Nmap']
    image_rocaucs_df = pd.DataFrame(MODALITY_NAMES, columns=['Method'])
    pixel_rocaucs_df = pd.DataFrame(MODALITY_NAMES, columns=['Method'])
    au_pros_df = pd.DataFrame(MODALITY_NAMES, columns=['Method'])

    for cls in classes:
        runner = Runner(args, cls, MODALITY_NAMES)
        if "memory" in args.method_name:
            runner.fit()
            # runner.alignment()
        image_rocaucs, pixel_rocaucs, au_pros, rec_loss = runner.evaluate()
        image_rocaucs_df[cls.title()] = image_rocaucs_df['Method'].map(image_rocaucs)
        pixel_rocaucs_df[cls.title()] = pixel_rocaucs_df['Method'].map(pixel_rocaucs)
        au_pros_df[cls.title()] = au_pros_df['Method'].map(au_pros)
        print(f"\nFinished running on class {cls}")
        print("################################################################################\n\n")

    image_rocaucs_df['Mean'] = round(image_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    pixel_rocaucs_df['Mean'] = round(pixel_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    au_pros_df['Mean'] = round(au_pros_df.iloc[:, 1:].mean(axis=1),3)
   
    print("\n\n################################################################################")
    print("############################# Image ROCAUC Results #############################")
    print("################################################################################\n")
    print(image_rocaucs_df.to_markdown(index=False))
    result_file.write(f'Image ROCAUC Results \n\n{image_rocaucs_df.to_markdown(index=False)} \n\n')

    print("\n\n################################################################################")
    print("############################# Pixel ROCAUC Results #############################")
    print("################################################################################\n")
    print(pixel_rocaucs_df.to_markdown(index=False))
    result_file.write(f'Pixel ROCAUC Results \n\n{pixel_rocaucs_df.to_markdown(index=False)} \n\n')

    print("\n\n################################################################################")
    print("################################ AU PRO Results ################################")
    print("################################################################################\n")
    print(au_pros_df.to_markdown(index=False))
    result_file.write(f'AU PRO Results \n\n{au_pros_df.to_markdown(index=False)} \n\n')
    result_file.close()

if __name__ == "__main__":    
    for i in test_t:
        args = parser.parse_args()
        args.noise_intensity = i
        if DEBUG == True:
            FILE_NAME = "Testing"
        else:
            FILE_NAME = f"_{args.method_name}_noiseT{args.noise_intensity}_Step{args.step_size}_Layers{args.feature_layers}_Topk{args.topk}_V1-5AugApply"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.output_dir = os.path.join(args.output_dir, args.dataset_type, time) + FILE_NAME
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        log_args(args)
        print("current device", device)
        run(args)
