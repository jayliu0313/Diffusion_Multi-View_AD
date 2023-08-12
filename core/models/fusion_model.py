import torch

class FusionModel(torch.nn.modules):
    def __init__(self, device, rgb_backbone_name='vit_base_patch8_224_dino', out_indices=None, checkpoint_path='',
                 pool_last=False, xyz_backbone_name='Point_MAE', group_size=128, num_group=1024):
        super(FusionModel, self).__init__()
