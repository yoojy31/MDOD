from .res_fpn import ResNet18FPN, ResNet34FPN, ResNet50FPN, ResNet101FPN


def get_ft_extractor(ft_extractor_key):
    return {
        'res18fpn': ResNet18FPN,
        'res34fpn': ResNet34FPN,
        'res50fpn': ResNet50FPN,
        'res101fpn': ResNet101FPN,
        None: None
    }[ft_extractor_key]
