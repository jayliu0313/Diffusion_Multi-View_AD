import timm
import torch
import torch.nn as nn
from core.models.initializer import initialize_from_cfg


def conv3x3(inplanes, outplanes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        inplanes,
        outplanes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(inplanes, outplanes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        shortcut=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride=1)
        self.upsample = None
        if stride != 1:
            self.upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.shortcut = shortcut
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.upsample is not None:
            out = self.upsample(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        upsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.upsample layers upsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out

class RGB_Extractor(torch.nn.Module):
    def __init__(self, device, backbone_name='resnet34', out_indices=(1, 2, 3), checkpoint_path='',
                 pool_last=False):
        super(RGB_Extractor, self).__init__()
        # Determine if to output features.
        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})

        self.backbone = timm.create_model(model_name=backbone_name, pretrained=True, checkpoint_path=checkpoint_path,
                                          **kwargs)
        self.device = device
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1)) if pool_last else None

    def forward(self, x):
        with torch.no_grad():
            x = x.to(self.device)

            # Backbone forward pass.
            features = self.backbone(x)

            # Adaptive average pool over the last layer.
            if self.avg_pool:
                fmap = features[-1]
                fmap = self.avg_pool(fmap)
                fmap = torch.flatten(fmap, 1)
                features.append(fmap)
        
        return features
    
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