import abc


class TesterABC(abc.ABC):
    def __init__(self, global_args, tester_args):
        self.global_args = global_args
        self.tester_args = tester_args

    @ abc.abstractmethod
    def run(self, framework, data_loader, result_dir):
        pass
