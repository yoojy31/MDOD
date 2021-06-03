import abc
import torch.nn as nn


class FTExtractorABC(abc.ABC, nn.Module):
    def __init__(self, global_args, ft_extractor_args):
        super(FTExtractorABC, self).__init__()
        self.global_args = global_args
        self.ft_extractor_args = ft_extractor_args
        self.pretrained = ft_extractor_args['pretrained']
        self.net = nn.ModuleDict()

    @ abc.abstractmethod
    def build(self):
        pass

    @ abc.abstractmethod
    def get_fmap2img_ratios(self):
        pass

    @ abc.abstractmethod
    def get_num_output_ch(self):
        pass
