import timm
import torch
import torch.nn as nn
from core.models.initializer import initialize_from_cfg

class RGB_Extractor(torch.nn.Module):
    def __init__(self, device, backbone_name='resnet34', out_indices=(1, 2, 3, 4), checkpoint_path='',
                 pool_last=False):          #efficientnet_b4, wide_resnet50_v2
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
  
class ResNet_Decoder(nn.Module):
    def __init__(
        self,
        inplanes,
        instrides,
        block,
        layers,
        groups=1,
        width_per_group=64,
        norm_layer=None,
        initializer=None,
    ):
        super(ResNet_Decoder, self).__init__()
        assert isinstance(inplanes, list) and len(inplanes) == 1
        assert isinstance(instrides, list) and len(instrides) == 1
        self.inplanes = inplanes[0]
        self.instrides = instrides[0]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        layer_planes = [64, 128, 256, 512]
        if self.instrides == 32:
            layer_strides = [2, 2, 2, 1]
        elif self.instrides == 16:
            layer_strides = [1, 2, 2, 1]
        else:
            raise NotImplementedError

        self.groups = groups
        self.base_width = width_per_group
        self.layer4 = self._make_layer(
            block, layer_planes[3], layers[3], stride=layer_strides[3]
        )
        self.layer3 = self._make_layer(
            block, layer_planes[2], layers[2], stride=layer_strides[2]
        )
        self.layer2 = self._make_layer(
            block, layer_planes[1], layers[1], stride=layer_strides[1]
        )
        self.layer1 = self._make_layer(
            block, layer_planes[1], layers[0], stride=layer_strides[0]
        )
        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv1 = nn.Conv2d(
            self.inplanes, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv2 = nn.Conv2d(self.inplanes, 3, kernel_size=1, stride=1, bias=False)
        initialize_from_cfg(self, initializer)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        shortcut = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * 1:
            shortcut = nn.Sequential(
                conv1x1(self.inplanes, planes * 1, stride=1),
                nn.Upsample(scale_factor=stride, mode="bilinear"),
                norm_layer(planes * 1),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                shortcut,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * 1
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    @property
    def layer0(self):
        return nn.Sequential(
            self.upsample1, self.conv1, self.bn1, self.relu, self.upsample2, self.conv2
        )

    def forward(self, input):
        x = input

        for layer_idx in range(4, -1, -1):
            layer = getattr(self, f"layer{layer_idx}", None)
            if layer is not None:
                x = layer(x)

        return {"image_rec": x}
