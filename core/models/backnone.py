import timm
import torch
import torch.nn as nn
from core.models.initializer import initialize_from_cfg

class RGB_Extractor(torch.nn.Module):
    def __init__(self, device, backbone_name='vit_base_patch8_224_dino', out_indices=None, checkpoint_path='',
                 pool_last=False):          #efficientnet_b4, wide_resnet50_v2
        super(RGB_Extractor, self).__init__()
        # Determine if to output features.
        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})

        self.backbone = timm.create_model(model_name=backbone_name, pretrained=True, checkpoint_path=checkpoint_path,
                                          **kwargs)
   
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1)) if pool_last else None
        
    def forward(self, x):
        B, C, H, W = x.size()
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.norm_pre(x)
        if self.backbone.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        feat = x[:,1:].permute(0, 2, 1).view(B, -1, 28, 28)
        return feat
    
    # def forward(self, x):
    
    #     x = x.to(self.device)

    #     # Backbone forward pass.
    #     features = self.backbone(x)

    #     # Adaptive average pool over the last layer.
    #     if self.avg_pool:
    #         fmap = features[-1]
    #         fmap = self.avg_pool(fmap)
    #         fmap = torch.flatten(fmap, 1)
    #         features.append(fmap)
        
    #     return features
    
    def freeze_parameters(self, layers, freeze_bn=False):
        """ Freeze resent parameters. The layers which are not indicated in the layers list are freeze. """

        layers = [str(layer) for layer in layers]
        # Freeze first block.
        if '1' not in layers:
            if hasattr(self.backbone, 'conv1'):
                for p in self.backbone.conv1.parameters():
                    p.requires_grad = False
            if hasattr(self.backbone, 'bn1'):
                for p in self.backbone.bn1.parameters():
                    p.requires_grad = False
            if hasattr(self.backbone, 'layer1'):
                for p in self.backbone.layer1.parameters():
                    p.requires_grad = False

        # Freeze second block.
        if '2' not in layers:
            if hasattr(self.backbone, 'layer2'):
                for p in self.backbone.layer2.parameters():
                    p.requires_grad = False

        # Freeze third block.
        if '3' not in layers:
            if hasattr(self.backbone, 'layer3'):
                for p in self.backbone.layer3.parameters():
                    p.requires_grad = False

        # Freeze fourth block.
        if '4' not in layers:
            if hasattr(self.backbone, 'layer4'):
                for p in self.backbone.layer4.parameters():
                    p.requires_grad = False

        # Freeze last FC layer.
        if '-1' not in layers:
            if hasattr(self.backbone, 'fc'):
                for p in self.backbone.fc.parameters():
                    p.requires_grad = False

        if freeze_bn:
            for module in self.backbone.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()