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
parser.add_argument('--output_dir', default="./output")
parser.add_argument('--dataset_type', default="eyecandies")
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument("--workers", default=4)
parser.add_argument('--CUDA', type=int, default=0, help="choose the device of CUDA")
parser.add_argument('--viz', action="store_true")
parser.add_argument('--seed', type=int, default=7)

# Method choose
parser.add_argument('--method_name', default="controlnet_ddiminv_memory", help="controlnet_rec, ddim_rec, nullinv_rec, ddim_memory, ddiminvrgb_memory, ddiminvnmap_memory, ddiminvunified_memory, ddiminvunified_multimemory, controlnet_ddiminv_memory")
parser.add_argument('--score_type', default=0, type=int, help="0 is max score, 1 is mean score") # just for score map, max score: maximum each pixel of 6 score maps, mean score: mean of 6 score maps 
parser.add_argument('--dist_function', type=str, default='l2_dist', help='l2_dist, cosine')

#### Load Checkpoint ####
parser.add_argument("--load_vae_ckpt", default="")
# "/mnt/home_6T/public/jayliu0313/check_point/rgb_ckpt/train_VAE_stable-diffusion-v1-4_woDecomp_allcls/vae_best_ckpt.pth"
# "/mnt/home_6T/public/jayliu0313/check_point/rgb_ckpt/vae_stable-diffusion-v1-4_woDecomp/vae_decomp_best_ckpt.pth"

parser.add_argument("--load_unet_ckpt", default="/mnt/home_6T/public/jayliu0313/check_point/Diffusion_ckpt/TrainUNet_ClsText_FeatureLossAllLayer_AllCls_epoch130/best_unet.pth")
# "/mnt/home_6T/public/jayliu0313/check_point/Diffusion_ckpt/TrainUNet_NullText_FeatureLossAllLayer_AllCls/best_unet.pth"
# "/mnt/home_6T/public/jayliu0313/check_point/Diffusion_ckpt/TrainUNet_ClsText_FeatureLossAllLayer_AllCls_epoch130/best_unet.pth"
# "/mnt/home_6T/public/jayliu0313/check_point/Diffusion_ckpt/TrainUnifiedUNet_ClsText_FeatureLossAllLayer_allcls/best_unet.pth"

parser.add_argument('--load_controlnet_ckpt', type=str, default="checkpoints/controlnet_model/NmapControlnet_RgbUnet_ClsTxt_allcls/controlnet_best_ckpt.pth")
# "/home/samchu0218/Multi_Lightings/checkpoints/controlnet_model/RgbNmap_UnetFLoss_AllClass_AllLayer_clsPrompt/controlnet_best.pth"

# Unet Model (Diffusion Model)
parser.add_argument("--diffusion_id", type=str, default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--revision", type=str, default="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c")

parser.add_argument("--noise_intensity", type=int, default=81) 
parser.add_argument("--memory_T", type=int, default=21)  # T
parser.add_argument("--memory_t", type=int, default=21)  # t
parser.add_argument("--test_T", type=int, default=21)    # T
parser.add_argument("--test_t", type=int, default=21)    # t
parser.add_argument("--step_size", type=int, default=20)

# DDIM Inv Setup
parser.add_argument("--opt_max_steps", type=int, default=1000)
parser.add_argument("--guidance_scale", type=float, default=7.5)

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
    FILE_NAME = f"_{args.method_name}_memoryT{args.memory_T}_memoryt{args.memory_t}_testT{args.test_T}_testt{args.test_t}_MULADD_WOAlign_ALLCLS"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
args.output_dir = os.path.join(args.output_dir, time) + FILE_NAME
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    

def run_eyecandies(args):
    if args.dataset_type=='eyecandies':
        classes = [
        'CandyCane',
        'ChocolateCookie',
        'ChocolatePraline',
        'Confetto',
        'GummyBear',
        'HazelnutTruffle',
        'LicoriceSandwich',
        'Lollipop',
        'Marshmallow',
        'PeppermintCandy'
        ]
    elif args.dataset_type=='mvtec3d':
        classes = []

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

log_args(args)
print("current device", device)
run_eyecandies(args)
