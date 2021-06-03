import abc
from torch.utils.data.dataset import Dataset
from .pre_proc import get_pre_proc


class DatasetABC(abc.ABC, Dataset):
    def __init__(self, global_args, dataset_args):
        super(DatasetABC, self).__init__()
        self.global_args = global_args
        self.roots = dataset_args['roots']
        self.types = dataset_args['types']
        pre_proc_key = dataset_args['pre_proc']
        pre_proc_args = dataset_args['pre_proc_args']
        self.pre_proc = get_pre_proc(pre_proc_key)(global_args, pre_proc_args)

    # @abc.abstractmethod
    # def shuffle(self):
    #     pass

    @ abc.abstractmethod
    def get_name2number_map(self):
        pass

    @ abc.abstractmethod
    def get_number2name_map(self):
        pass

    @ abc.abstractmethod
    def get_dataset_roots(self):
        pass
