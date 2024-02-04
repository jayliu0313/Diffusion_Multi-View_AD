import torch
import argparse
import os
import os.path as osp
import datetime
# from core.models.unet_model import UNet
from core.runner import Runner
import datetime
import pandas as pd
from utils.utils import set_seeds, log_args
os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser(description='test')

# Dataset and environment setup
parser.add_argument('--data_path', default="/mnt/home_6T/public/jayliu0313/datasets/Eyecandies/", type=str)
parser.add_argument('--output_dir', default="./output")
parser.add_argument('--dataset_type', default="eyecandies")
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument("--workers", default=4)
parser.add_argument('--CUDA', type=int, default=0, help="choose the device of CUDA")
parser.add_argument('--viz', action="store_true")
parser.add_argument('--seed', type=int, default=7)

# Method choose
parser.add_argument('--method_name', default="ddiminvunified_memory", help="controlnet_rec, ddim_rec, nullinv_rec, ddim_memory, ddiminvrgb_memory, ddiminvnmap_memory, ddiminvrgbnmap_memory, ddiminvunified_memory, controlnet_ddiminv_memory, directinv_memory, controlnet_directinv_memory")
parser.add_argument('--score_type', default=0, type=int, help="0 is max score, 1 is mean score") # just for score map, max score: maximum each pixel of 6 score maps, mean score: mean of 6 score maps 

#### Load Checkpoint ####
parser.add_argument("--load_vae_ckpt", default=None)
# "/mnt/home_6T/public/jayliu0313/check_point/rgb_ckpt/train_VAE_stable-diffusion-v1-4_woDecomp_allcls/vae_best_ckpt.pth"
# "/mnt/home_6T/public/jayliu0313/check_point/rgb_ckpt/vae_stable-diffusion-v1-4_woDecomp/vae_decomp_best_ckpt.pth"

parser.add_argument("--load_unet_ckpt", default="/mnt/home_6T/public/jayliu0313/check_point/Diffusion_ckpt/TrainUnifiedUNet_ClsText_FeatureLossAllLayer_3cls/best_unet.pth")
# "/mnt/home_6T/public/jayliu0313/check_point/Diffusion_ckpt/TrainUNet_NullText_FeatureLossAllLayer_AllCls/best_unet.pth"
# "/mnt/home_6T/public/jayliu0313/check_point/Diffusion_ckpt/TrainUNet_ClsText_FeatureLossAllLayer_AllCls_epoch130/best_unet.pth"
# "/mnt/home_6T/public/jayliu0313/check_point/Diffusion_ckpt/TrainUNet_ClsText_FeatureLossAllLayer_AllCls/best_unet.pth"
# "checkpoints/diffusion_checkpoints/TrainNmapUNet_ClsText_FeatureLossAllLayer_AllCls/best_unet.pth"
# "/mnt/home_6T/public/jayliu0313/check_point/Diffusion_ckpt/TrainRGBNmapUNet_ClsText_FeatureLossAllLayer_AllCls/best_unet.pth"
# "/mnt/home_6T/public/jayliu0313/check_point/Diffusion_ckpt/TrainUnifiedUNet_ClsText_FeatureLossAllLayer_3cls/best_unet.pth"

parser.add_argument('--load_controlnet_ckpt', type=str, default=None)
# "/home/samchu0218/Multi_Lightings/checkpoints/controlnet_model/RgbNmap_UnetFLoss_AllClass_AllLayer_clsPrompt/controlnet_best.pth"


parser.add_argument("--load_backbone_ckpt", default=None)
parser.add_argument('--load_nmap_ckpt_path', default=None)

parser.add_argument('--backbone_name', default="vit_base_patch8_224_dino")
# Unet Model (Diffusion Model)
parser.add_argument("--diffusion_id", type=str, default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--revision", type=str, default="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c")
parser.add_argument("--noise_intensity", type=int, default=81) 

parser.add_argument("--memory_T", type=int, default=81)  # T
parser.add_argument("--memory_t", type=int, default=81)  # t
parser.add_argument("--test_T", type=int, default=81)  # T
parser.add_argument("--test_t", type=int, default=81)  # t

parser.add_argument("--step_size", type=int, default=20)


# DDIM Inv Setup
parser.add_argument("--opt_max_steps", type=int, default=1000)
parser.add_argument("--num_opt_steps", type=int, default=3)
parser.add_argument("--guidance_scale", type=float, default=7.5)

# 1000, 20, 7.5, 3
DEBUG = False
# Controlnet Model Setup
parser.add_argument("--controllora_linear_rank", type=int, default=4)
parser.add_argument("--controllora_conv2d_rank", type=int, default=0)

args = parser.parse_args()
if DEBUG == True:
    FILE_NAME = "Testing"
else:
    # FILE_NAME = "ddiminv_unet_4thlayers_noise1_textpromptnormal"
    # FILE_NAME = f"_{args.method_name}_noise{args.noise_intensity}_step{args.step_size}_loop{args.num_opt_steps}_gdscale{args.guidance_scale}_clsprompt"
    # FILE_NAME = f"_{args.method_name}_noise{args.noise_intensity}_step{args.step_size}_memory{args.memory_intensity}_FeatureLoss_clstxt_method1"
    FILE_NAME = f"_{args.method_name}_memoryT{args.memory_T}_memoryt{args.memory_t}_testT{args.test_T}_testt{args.test_t}_FeatureLoss_clstxt_DualMemory"
cuda_idx = str(args.CUDA)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_idx
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("current device", device)
time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
args.output_dir = os.path.join(args.output_dir, time) + FILE_NAME
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    
log_args(args)

def run_eyecandies(args):
    if args.dataset_type=='eyecandies':
        classes = [
        'CandyCane',
        'ChocolateCookie',
        # 'ChocolatePraline',
        # 'Confetto',
        'GummyBear',
        # 'HazelnutTruffle',
        # 'LicoriceSandwich',
        # 'Lollipop',
        # 'Marshmallow',
        # 'PeppermintCandy'
        ]
    elif args.dataset_type=='mvtec3d':
        classes = []

    result_file = open(osp.join(args.output_dir, "results.txt"), "a", 1)    
    METHOD_NAMES = [args.method_name]
    image_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    pixel_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    au_pros_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    rec_loss_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    for cls in classes:
        runner = Runner(args, cls)
        if "memory" in args.method_name:
            runner.fit()
        image_rocaucs, pixel_rocaucs, au_pros, rec_loss = runner.evaluate()
        image_rocaucs_df[cls.title()] = image_rocaucs_df['Method'].map(image_rocaucs)
        pixel_rocaucs_df[cls.title()] = pixel_rocaucs_df['Method'].map(pixel_rocaucs)
        au_pros_df[cls.title()] = au_pros_df['Method'].map(au_pros)
        rec_loss_df[cls.title()] = rec_loss_df['Method'].map(rec_loss)
        print(f"\nFinished running on class {cls}")
        print("################################################################################\n\n")

    image_rocaucs_df['Mean'] = round(image_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    pixel_rocaucs_df['Mean'] = round(pixel_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    au_pros_df['Mean'] = round(au_pros_df.iloc[:, 1:].mean(axis=1),3)
    rec_loss_df['Mean'] = round(rec_loss_df.iloc[:, 1:].mean(axis=1),6)

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

    print("\n\n##########################################################################")
    print("############################# AU PRO Results #############################")
    print("##########################################################################\n")
    print(au_pros_df.to_markdown(index=False))
    result_file.write(f'AU PRO Results \n\n{au_pros_df.to_markdown(index=False)} \n\n')

    print("\n\n##########################################################################")
    print("############################# MSE Loss Results #############################")
    print("##########################################################################\n")
    print(rec_loss_df.to_markdown(index=False))
    result_file.write(f'Reconstruction Loss Results \n\n{rec_loss_df.to_markdown(index=False)}')
    
    result_file.close()


run_eyecandies(args)
