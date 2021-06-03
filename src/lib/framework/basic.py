import time
import torch
import torch.distributed
from .__abc__ import FrameworkABC


class BasicFramework(FrameworkABC):
    def forward(self, data_dict, train=True, grad_enable=True):
        self.network.train(train)
        torch.autograd.set_grad_enabled(grad_enable)

        data_dict['img'] = data_dict['img'].requires_grad_(grad_enable).float().cuda()

        if train:
            data_dict['boxes'] = data_dict['boxes'].cuda()
            data_dict['labels'] = data_dict['labels'].long().cuda()
            data_dict['n_boxes'] = data_dict['n_boxes'].long().cuda()
            output_dict, loss_dict, value_dict = self.network.forward(data_dict, loss=True)

            loss_dict = self.merge_batch_losses(loss_dict)
            value_dict = self.merge_batch_values(value_dict)
            return output_dict, loss_dict, value_dict

        else:
            t1 = time.time()
            output_dict = self.network.forward(data_dict, loss=False)
            t2 = time.time()
            result_dict, value_dict = self.post_proc.process(output_dict, data_dict)
            t3 = time.time()
            infer_time, net_time, pp_time = t3 - t1, t2 - t1, t3 - t2

            value_dict = self.merge_batch_values(value_dict)
            value_dict.update({'infer_time': infer_time, 'net_time': net_time, 'pp_time': pp_time})
            return output_dict, result_dict, value_dict

    def merge_batch_losses(self, loss_dict):
        batch_loss_dict = dict()
        for key, value in loss_dict.items():
            if self.world_size > 1:
                gather_num = [torch.ones(1).long().cuda() for _ in range(self.world_size)]
                torch.distributed.all_gather(gather_num, torch.tensor(value.shape[0]).view(1).long().cuda())
                # gather_sum = [torch.ones(1).cuda() for _ in range(self.world_size)]
                # torch.distributed.all_gather(gather_sum, torch.sum(value).view(1))

                if len(value) > 0:
                    batch_loss_dict[key] = \
                        self.world_size * torch.sum(value, dim=0) / torch.sum(torch.cat(gather_num, dim=0))
                    # print(key, torch.sum(value), gather_sum, value.shape[0], gather_num)

            else:
                batch_loss_dict[key] = torch.sum(value, dim=0) / value.shape[0]
            # if 'mog_nll' == key:
            #     batch_loss_dict[key] = torch.sum(value, dim=0) / (5.5 * 4)
            # elif 'mod_nll' == key:
            #     batch_loss_dict[key] = torch.sum(value, dim=0) / (5.5 * 2 * 32)
            # else:
            #     raise TypeError
        return batch_loss_dict

    def merge_batch_values(self, value_dict):
        batch_value_dict = dict()
        for key, value in value_dict.items():
            if len(value) > 0:
                batch_value_dict[key] = torch.sum(value, dim=0) / value.shape[0]
            else:
                batch_value_dict[key] = None
        return batch_value_dict
