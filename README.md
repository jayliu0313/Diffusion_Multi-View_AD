# Learning Diffusion Models for Multi-View Anomaly Detection (ECCV2024)
### [Paper Link](https://eccv.ecva.net/virtual/2024/poster/1911)

## Installation
### Requirement
Linux (Ubuntu 16.04)  
Python 3.9+  
PyTorch 1.7 or higher  
CUDA 10.2 or higher

### create environment
```
git clone https://github.com/jayliu0313/Multi_Lightings.git
cd Multi_Lightings
conda create --name myenv python=3.6
conda activate myenv
pip install -r requirements.txt
```

## Eyecandies Dataset
[Here](https://eyecan-ai.github.io/eyecandies/download) to download dataset

## Eyecandies Implementation
### Finetune the UNet and CountrolNet
```
python train_unet.py --data_path DATASET_PATH --ckpt_path SAVE_PATH
```
```
python train_controlnet.py --data_path DATASET_PATH --load_unet_ckpt UNET_CKPT_PATH --ckpt_path SAVE_PATH
```

### Buid Memory and Inference
The result will be stored in the output directory. <br />
You can use "--vis" to visualize our result of the heat map. 
```
python test.py --datasets_path DATASET_PATH --load_unet_ckpt UNET_CKPT_PATH --load_controlnet_ckpt CONTROLNET_CKPT_PATH
```

---------------------------------------------------------------------------------------
We also provide another dataset MVTec3D-AD, which demonstrates the generalizability of our approach.
*Note: due to the lack of multi-view RGB images in the MVTec 3D-AD dataset, our method does not use Feature Loss.*
## MvTec3D-AD Dataset
[Here](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) to download dataset

## MvTec3D-AD Implementation
### Preprocessing
In MVTec3D_preprocess.py, you need to change the variable ```DATASET_PATH```, which is the root directory of the MVTec 3D-AD dataset; and also change the variable ```RESULT_PATH```, which is the target location where the dataset is stored after preprocessing. <br />
After setting, you can run and wait few minutes:
```
python MVTec3D_preprocess.py
```

### Finetune the UNet and CountrolNet
```
python train_unet.py --data_path [DATASET_PATH] --ckpt_path [SAVE_PATH] --dataset_type mvtec3d --diffusion_id runwayml/stable-diffusion-v1-5
```
```
python train_controlnet.py --data_path [PREPROCESS_DATASET_PATH]--load_unet_ckpt [UNET_CKPT_PATH] --ckpt_path [SAVE_PATH] --dataset_type mvtec3d
```

### Buid Memory and Inference
The result will be stored in the output directory. <br />
You can use "--vis" to visualize our result of the heat map. 
```
python test.py --datasets_path [DATASET_PATH] --load_unet_ckpt [UNET_CKPT_PATH] --load_controlnet_ckpt [CONTROLNET_CKPT_PATH] --dataset_type mvtec3d --topk 3 --rgb_weight 0.25 --nmap_weight 0.75 --noise_intensity 41 --feature_layers 2 --feature_layers 3
```

## Reference
Our training pipeline is refer to https://github.com/johannakarras/DreamPose. <br />
Our memory architecture is refer to https://github.com/jayliu0313/Shape-Guided. <br />
During testing, we referred to the strategy of Null-Text Inversion: https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images. <br />