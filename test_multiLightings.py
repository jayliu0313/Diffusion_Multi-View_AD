import torch
import argparse
import numpy as np
import os
import os.path as osp
import datetime

from core.models.network import Convolution_AE#, Autoencoder
# from core.models.unet_model import UNet
# from core.models.mae import mae_vit_base_patch16_dec512d8b
from core.runner import Runner
import datetime
import pandas as pd

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--data_path', default="/mnt/home_6T/public/jayliu0313/datasets/Eyecandies/", type=str)
parser.add_argument('--ckpt_path', default="checkpoints/cnn_meanRec_lr00003/ckpt_000599.pth")
parser.add_argument('--output_dir', default="./output")
parser.add_argument('--dataset_type', default="eyecandies")
parser.add_argument('--method_name', default="reconstruct", help="mean_reconstruct, reconstruct")
parser.add_argument('--score_type', default=0, type=int, help="0 is max score, 1 is mean score")
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument("--workers", default=8)
parser.add_argument('--CUDA', type=int, default=0, help="choose the device of CUDA")

args = parser.parse_args()
cuda_idx = str(args.CUDA)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_idx
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("current device", device)
time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
args.output_dir = os.path.join(args.output_dir, time)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

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

    # build model and load ckpt
    # model = Autoencoder(args, device).to(device)
    model = Convolution_AE(args, device)
    model.to(device)
    model.eval()
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    METHOD_NAMES = [args.method_name]
    image_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    pixel_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    au_pros_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    rec_loss_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    for cls in classes:
        runner = Runner(args, model, cls)
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
    print("############################# AU PRO Results #############################")
    print("##########################################################################\n")
    print(rec_loss_df.to_markdown(index=False))
    result_file.write(f'Reconstruction Loss Results \n\n{rec_loss_df.to_markdown(index=False)}')
    
    result_file.close()


run_eyecandies(args)            
