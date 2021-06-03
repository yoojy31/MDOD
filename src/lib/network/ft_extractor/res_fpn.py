import torchvision
import torch.nn as nn
from torchvision.models.resnet import Bottleneck
from .. import __util__ as net_util, __module__
from .__abc__ import FTExtractorABC


class ResNet34FPN(FTExtractorABC):
    def __init__(self, global_args, ft_extractor_args):
        super(ResNet34FPN, self).__init__(global_args, ft_extractor_args)
        self.inter_chs = (128, 256, 512)
        self.num_filters = ft_extractor_args['num_filters']
        self.upsampling = ft_extractor_args['upsampling']
        self.fpn_actv_func = __module__.get_actv_func(ft_extractor_args['fpn_actv_func'])
        self.group_norm_size = ft_extractor_args['group_norm_size']
        self.group_norm = ft_extractor_args['group_norm']
        self.init_method = ft_extractor_args['init_method']

    def __get_base_network__(self):
        return torchvision.models.resnet34(pretrained=self.pretrained)

    def get_fmap2img_ratios(self):
        return 1/8.0, 1/16.0, 1/32.0, 1/64.0, 1/128.0

    def get_num_output_ch(self):
        return self.num_filters

    def build(self):
        base_net = self.__get_base_network__()
        self.net['base'] = nn.Sequential(
            base_net.conv1, base_net.bn1, base_net.relu,
            base_net.maxpool, base_net.layer1)
        self.net['stage_c3'] = base_net.layer2
        self.net['stage_c4'] = base_net.layer3
        self.net['stage_c5'] = base_net.layer4
        # self.net['fpn'] = __module__.Level5FPN(
        #     self.inter_chs, self.num_filters, self.upsampling, self.fpn_actv_func)
        self.net['fpn'] = __module__.VarLevel5FPN(
            self.inter_chs, self.num_filters, self.upsampling, self.fpn_actv_func)

        if self.group_norm:
            name_module_list = list(self.net.named_modules())
            for name, module in name_module_list:
                if isinstance(module, nn.BatchNorm2d):
                    parent_module = net_util.get_parent_module(self.net, name)
                    gn = nn.GroupNorm(self.group_norm_size, module.num_features)

                    key = name.split('.')[-1]
                    if isinstance(parent_module, Bottleneck):
                        if key == 'bn1':
                            parent_module.bn1 = gn
                        elif key == 'bn2':
                            parent_module.bn2 = gn
                        elif key == 'bn3':
                            parent_module.bn3 = gn
                        elif key == 'bn':
                            parent_module.bn = gn
                    else:
                        parent_module[int(key) if key.isdigit() else key] = gn

        if self.pretrained:
            print('[BACKBONE] load image-net pre-trained model')
        else:
            net_util.init_modules(
                [self.net['base'], self.net['stage_c3'],
                 self.net['stage_c4'], self.net['stage_c5']], self.init_method)
        net_util.init_modules(self.net['fpn'].modules(), self.init_method)

    def forward(self, image):
        # print(self.net)
        # exit()
        base_fmap = self.net['base'].forward(image)
        fmap_c3 = self.net['stage_c3'].forward(base_fmap)
        fmap_c4 = self.net['stage_c4'].forward(fmap_c3)
        fmap_c5 = self.net['stage_c5'].forward(fmap_c4)
        fmaps = self.net['fpn'].forward([fmap_c3, fmap_c4, fmap_c5])
        return fmaps


class ResNet18FPN(ResNet34FPN):
    def __init__(self, global_args, ft_extractor_args):
        super(ResNet18FPN, self).__init__(global_args, ft_extractor_args)
        self.inter_chs = (128, 256, 512)

    def __get_base_network__(self):
        return torchvision.models.resnet18(pretrained=self.pretrained)


class ResNet50FPN(ResNet34FPN):
    def __init__(self, global_args, ft_extractor_args):
        super(ResNet50FPN, self).__init__(global_args, ft_extractor_args)
        self.inter_chs = (512, 1024, 2048)

    def __get_base_network__(self):
        return torchvision.models.resnet50(pretrained=self.pretrained)


class ResNet101FPN(ResNet50FPN):
    def __get_base_network__(self):
        return torchvision.models.resnet101(pretrained=self.pretrained)

# [ResNet-18]
# ResNet(
#   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (layer1): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer2): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer3): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer4): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (fc): Linear(in_features=512, out_features=1000, bias=True)
# )

# [ResNet-50]
# ResNet(
#   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (layer1): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsample): Sequential(
#         (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#   )
#   (layer2): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsample): Sequential(
#         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (3): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#   )
#   (layer3): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsample): Sequential(
#         (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (3): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (4): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (5): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#   )
#   (layer4): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsample): Sequential(
#         (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (fc): Linear(in_features=2048, out_features=1000, bias=True)
# )