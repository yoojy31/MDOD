import abc
import torch
import torch.nn as nn
from .ft_extractor import get_ft_extractor
from .loss_func import get_loss_func


class NetworkABC(abc.ABC, nn.Module):
    def __init__(self, global_args, network_args):
        super(NetworkABC, self).__init__()
        self.global_args = global_args
        self.network_args = network_args
        if 'ft_extractor' in network_args.keys():
            ft_extractor_key = network_args['ft_extractor']
            ft_extractor_args = network_args['ft_extractor_args']
            self.ft_extractor = get_ft_extractor(ft_extractor_key)(global_args, ft_extractor_args)

        if 'loss_func' in network_args.keys():
            loss_func_key = network_args['loss_func']
            loss_func_args = network_args['loss_func_args']
            self.loss_func = get_loss_func(loss_func_key)(global_args, loss_func_args)
        self.net = nn.ModuleDict()

    @ abc.abstractmethod
    def build(self):
        pass

    def save(self, save_path):
        if self.net is not None:
            torch.save(self.net.state_dict(), save_path)
            print('[NETWORK] save: %s' % save_path)

    def load(self, load_path):
        if self.net is not None:
            net_dict = torch.load(load_path, map_location='cpu')
            self.net.load_state_dict(net_dict)
            print('[NETWORK] load: %s' % load_path)