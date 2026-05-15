import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class TimmResNet50(nn.Module):
    def __init__(self, pretrained=None, with_cp=False, out_indices=(0, 1, 2, 3)):
        super().__init__()
        model = timm.create_model(
            "resnet50",
            features_only=False,
            pretrained=False,
            output_stride=32,
        )
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.out_indices = out_indices
        self.with_cp = with_cp
        if pretrained:
            self.load_torchvision_resnet50(pretrained)

    def load_torchvision_resnet50(self, path):
        state = torch.load(path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.load_state_dict(state, strict=False)

    def forward(self, x):
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        for i, layer in enumerate((self.layer1, self.layer2, self.layer3, self.layer4)):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)


class FPN(nn.Module):
    def __init__(
        self,
        in_channels=(256, 512, 1024, 2048),
        out_channels=256,
        num_outs=5,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channel in in_channels:
            self.lateral_convs.append(ConvModule(in_channel, out_channels, 1))
            self.fpn_convs.append(ConvModule(out_channels, out_channels, 3, padding=1))
        extra_levels = num_outs - len(in_channels)
        for _ in range(extra_levels):
            self.fpn_convs.append(ConvModule(out_channels, out_channels, 3, stride=2, padding=1))

    def forward(self, inputs):
        laterals = [conv(inputs[i]) for i, conv in enumerate(self.lateral_convs)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        while len(outs) < self.num_outs:
            source = outs[-1]
            if self.relu_before_extra_convs:
                source = F.relu(source)
            outs.append(self.fpn_convs[len(outs)](source))
        return tuple(outs)


class ResNet50FPN(nn.Module):
    def __init__(self, pretrained=None, out_channels=256, num_outs=5, in_channels=(256, 512, 1024, 2048), with_cp=False):
        super().__init__()
        self.backbone = TimmResNet50(pretrained=pretrained, with_cp=with_cp)
        self.neck = FPN(in_channels=in_channels, out_channels=out_channels, num_outs=num_outs)

    def forward(self, x):
        return self.neck(self.backbone(x))
