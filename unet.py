import torch
import torch.nn as nn
from torchvision import models

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetUNet(nn.Module):
    def __init__(self, n_class, channels=1, base_width=64, dataset="mnist"):
        super().__init__()
        self.dilation = 1
        self.inplanes = base_width
        self.channels = channels
        self.groups = 1
        self.base_width = base_width
        self._norm_layer = nn.BatchNorm2d
        if dataset == "mnist":
            self.img_dim = 28
        elif dataset == "cifar":
            self.img_dim = 32
        else:
            assert False

        self.layer0 = nn.Sequential(
            nn.Conv2d(channels, self.base_width, 7, stride=2, padding=3),
            nn.BatchNorm2d(self.base_width),
            nn.ReLU()
        ) # size=(N, 64, x.H/2, x.W/2) 16*16

        self.layer0_1x1 = convrelu(self.base_width, self.base_width, 1, 0)
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock, self.base_width, 2)
        )  # size=(N, 64, x.H/4, x.W/4) 8*8
        self.layer1_1x1 = convrelu(self.base_width, self.base_width, 1, 0)
        self.layer2 = self._make_layer(BasicBlock, self.base_width*2, 2, stride=2,
                                       dilate=False)  # size=(N, 128, x.H/8, x.W/8) 4*4
        self.layer2_1x1 = convrelu(self.base_width*2, self.base_width*2, 1, 0)
        self.layer3 = self._make_layer(BasicBlock, self.base_width*4, 2, stride=2,
                                       dilate=False)  # size=(N, 256, x.H/2, x.W/2) 2*2
        self.layer3_1x1 = convrelu(self.base_width*4, self.base_width*4, 1, 0)

        self.upsample1 = nn.Upsample(((self.img_dim-1)//8+1, (self.img_dim-1)//8+1), mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(((self.img_dim-1)//4+1, (self.img_dim-1)//4+1), mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(((self.img_dim-1)//2+1, (self.img_dim-1)//2+1), mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample((self.img_dim, self.img_dim), mode='bilinear', align_corners=True)

        self.conv_up2 = convrelu(self.base_width*(2+4), self.base_width*4, 3, 1)
        self.conv_up1 = convrelu(self.base_width*(1+4), self.base_width*2, 3, 1)
        self.conv_up0 = convrelu(self.base_width*(1+2), self.base_width*1, 3, 1)

        self.conv_original_size0 = convrelu(channels, self.base_width, 3, 1)
        self.conv_original_size1 = convrelu(self.base_width, self.base_width, 3, 1)
        self.conv_original_size2 = convrelu(self.base_width*(1+1), self.base_width, 3, 1)

        self.conv_last = nn.Conv2d(self.base_width, n_class, 1)

    def forward(self, input):
        # if self.channels == 1:
        #     input = nn.ConstantPad2d((0,4,0,4), 0)(input)
        x_original = self.conv_original_size0(input)  # MNIST (N, base_width, 28, 28)
        x_original = self.conv_original_size1(x_original)  # MNIST (N, base_width, 28, 28)

        layer0 = self.layer0(input)  # MNIST (N, base_width, 14, 14)
        layer1 = self.layer1(layer0) # MNIST (N, base_width, 7, 7)
        layer2 = self.layer2(layer1) # MNIST (N, base_width*2, 4, 4)
        layer3 = self.layer3(layer2) # MNIST (N, base_width*4, 2, 2)

        layer3 = self.layer3_1x1(layer3) # MNIST (N, base_width*4, 2, 2)
        x = self.upsample1(layer3) # MNIST (N, base_width*4, 4, 4)

        layer2 = self.layer2_1x1(layer2) # MNIST (N, base_width*2, 4, 4)
        x = torch.cat([x, layer2], dim=1) # MNIST (N, base_width*2 + base_width*4, 4, 4)

        x = self.conv_up2(x) # MNIST (N, base_width*4, 4, 4)
        x = self.upsample2(x) # MNIST (N, base_width*4, 7, 7)
        layer1 = self.layer1_1x1(layer1) # MNIST (N, base_width, 7, 7)
        x = torch.cat([x, layer1], dim=1) # MNIST (N, base_width, 7, 7)
        x = self.conv_up1(x) # MNIST (N, base_width*(1 + 4), 7, 7)

        x = self.upsample3(x) # MNIST (N, base_width*(1 + 4), 14, 14)
        layer0 = self.layer0_1x1(layer0)  # MNIST (N, base_width*2, 14, 14)
        x = torch.cat([x, layer0], dim=1) # MNIST (N, base_width*(2+1), 14, 14)
        x = self.conv_up0(x) # MNIST (N, base_width*(1), 14, 14)

        x = self.upsample4(x) # MNIST (N, base_width*(1), 28, 28)
        x = torch.cat([x, x_original], dim=1) # MNIST (N, base_width*(1+1) , 28, 28)
        x = self.conv_original_size2(x) # MNIST (N, base_width*(1) , 28, 28)

        out = self.conv_last(x) # MNIST (N, n_class , 28, 28)
        return out

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

