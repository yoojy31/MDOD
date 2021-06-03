import abc


class FrameworkABC(abc.ABC):
    def __init__(self, global_args, framework_args, network, post_proc, world_size):
        self.global_args = global_args
        self.framework_args = framework_args
        self.network = network
        self.post_proc = post_proc
        self.world_size = world_size

    def train_forward(self, data_dict):
        output_dict, loss_dict, value_dict = self.forward(data_dict, train=True, grad_enable=True)
        return output_dict, loss_dict, value_dict

    def infer_forward(self, data_dict):
        output_dict, result_dict, value_dict = self.forward(data_dict, train=False, grad_enable=False)
        return output_dict, result_dict, value_dict

    def valid_forward(self, data_dict):
        output_dict, loss_dict, value_dict = self.forward(data_dict, train=True, grad_enable=False)
        return output_dict, loss_dict, value_dict

    @ abc.abstractmethod
    def forward(self, data_dict, train=True, grad_enable=True):
        pass

    @ abc.abstractmethod
    def merge_batch_losses(self, loss_dict):
        pass

    @abc.abstractmethod
    def merge_batch_values(self, value_dict):
        pass


