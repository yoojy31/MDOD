import abc


class PostProcABC(abc.ABC):
    def __init__(self, global_args, post_proc_args):
        self.global_args = global_args
        self.post_proc_args = post_proc_args

    @ abc.abstractmethod
    def process(self, *x):
        pass
