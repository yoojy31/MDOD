import abc
from lib import util as lib_util


class LossFunctionABC(abc.ABC):
    def __init__(self, global_args, loss_func_args):
        self.global_args = global_args
        self.loss_func_args = loss_func_args
        self.lw_dict = loss_func_args['lw_dict']

    def update(self, new_loss_args):
        if 'lw_dict' in new_loss_args.keys():
            pre_lw_dict_str = str(self.lw_dict)
            self.lw_dict.update(new_loss_args['lw_dict'])
            print('[LOSS FUNCTION] lw dict:', pre_lw_dict_str, '->', self.lw_dict)

    @ abc.abstractmethod
    def forward(self, *x):
        pass
