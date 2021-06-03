import math
import random
import torch
import torch.nn as nn
import torchvision.ops as ops
from torch.nn import functional as func
from . import __module__
from lib import util as lib_util
from lib.network import __util__ as net_util
from .__abc__ import NetworkABC


class MDODNetwork(NetworkABC):
    def __init__(self, global_args, network_args):
        super(MDODNetwork, self).__init__(global_args, network_args)
        self.coord_scale_factor = global_args['coord_scale_factor']
        self.input_size = (global_args['img_h'], global_args['img_w'])
        self.coord_range = (
            self.input_size[0] / self.coord_scale_factor,
            self.input_size[1] / self.coord_scale_factor,
        )
        self.n_classes = global_args['n_classes']

        self.output_sizes = network_args['output_sizes']
        self.num_fmap_ch = self.ft_extractor.get_num_output_ch()
        self.num_filters = network_args['num_filters']
        self.std_factor = network_args['std_factor']

        self.split_idxes = [4, 4, self.n_classes, 1]
        self.output_ch = sum(self.split_idxes)
        self.max_batch_size = network_args['max_batch_size']

        self.center_offset = None
        self.output_scale = None
        self.limit_scale = None

    def build(self):
        self.ft_extractor.build()
        self.net['ft_extractor'] = self.ft_extractor
        self.net['detector'] = nn.Sequential(
            nn.Conv2d(self.num_fmap_ch, self.num_filters, 3, 1, 1, bias=True),
            __module__.get_actv_func(self.network_args['actv_func']),
            nn.Conv2d(self.num_filters, self.num_filters, 1, 1, 0, bias=True),
            __module__.get_actv_func(self.network_args['actv_func']),
            nn.Conv2d(self.num_filters, self.num_filters, 1, 1, 0, bias=True),
            __module__.get_actv_func(self.network_args['actv_func']),
            nn.Conv2d(self.num_filters, self.output_ch, 1, 1, 0, bias=True))

        output_sizes = list()
        for r in self.ft_extractor.get_fmap2img_ratios():
            f_map_h = math.ceil(self.input_size[0] * r)
            f_map_w = math.ceil(self.input_size[1] * r)
            output_sizes.append((f_map_h, f_map_w))

        self.center_offset = list()
        self.output_scale = list()
        self.limit_scale = list()

        for i, _ in enumerate(self.ft_extractor.get_fmap2img_ratios()):
            center_offset_i = torch.from_numpy(lib_util.create_coord_map(output_sizes[i], self.coord_range))
            center_offset_i = torch.cat([center_offset_i] * self.max_batch_size, dim=0)
            output_scale_i = torch.ones((self.max_batch_size, 4, output_sizes[i][0], output_sizes[i][1]))
            limit_scale_i = torch.ones(center_offset_i.shape)

            center_offset_i = center_offset_i.view(self.max_batch_size, 2, -1)
            output_scale_i = output_scale_i.view(self.max_batch_size, 4, -1) * 0.0625 * (2 ** i)
            limit_scale_i = limit_scale_i.view(self.max_batch_size, 2, -1) * (self.coord_range[0] / output_sizes[i][0])

            self.center_offset.append(center_offset_i)
            self.output_scale.append(output_scale_i)
            self.limit_scale.append(limit_scale_i)

        self.center_offset = torch.cat(self.center_offset, dim=2).cuda()
        self.output_scale = torch.cat(self.output_scale, dim=2).cuda()
        self.limit_scale = torch.cat(self.limit_scale, dim=2).cuda()

    def __get_output_tensors__(self, fmaps, net_data_dict, batch_size):
        out_tensors = list()
        for i, fmap in enumerate(fmaps):
            out_tensor = self.net['detector'].forward(fmap)
            out_tensors.append(out_tensor.view((batch_size, self.output_ch, -1)))

        out_tensor = torch.cat(out_tensors, dim=2)
        out_tensors = torch.split(out_tensor, self.split_idxes, dim=1)
        return out_tensors

    def __sync_batch_and_device__(self, batch_size, device_idx):
        if self.center_offset.device.index != device_idx:
            self.center_offset = self.center_offset.cuda(device_idx)
            self.output_scale = self.output_scale.cuda(device_idx)
            self.limit_scale = self.limit_scale.cuda(device_idx)

        center_offset = self.center_offset[:batch_size]
        output_scale = self.output_scale[:batch_size]
        limit_scale = self.limit_scale[:batch_size]
        return {'center_offset': center_offset, 'limit_scale': limit_scale, 'output_scale': output_scale}

    @ staticmethod
    def __decode__(o1, limit_scale, center_offset):
        o1_xy, o1_wh = torch.split(o1, [2, 2], dim=1)
        o1_xy = torch.tanh(o1_xy) * limit_scale + center_offset
        mu = net_util.__cvt_xywh2ltrb__(torch.cat([o1_xy, o1_wh], dim=1))
        return mu, o1_wh

    def __get_mixture_params__(self, out_tensors, net_data_dict):
        o1, o2, o3, o4 = out_tensors
        center_offset = net_data_dict['center_offset']
        output_scale = net_data_dict['output_scale']
        limit_scale = net_data_dict['limit_scale']

        mu, o1_wh = self.__decode__(o1 * output_scale, limit_scale, center_offset)
        o1_wh = torch.clamp_min(torch.cat([o1_wh, o1_wh], dim=1), lib_util.epsilon).detach()
        sig = torch.max(func.softplus(o2) * output_scale, o1_wh * self.std_factor)
        prob = func.softmax(o3, dim=1)
        pi = func.softmax(o4, dim=2)

        param_dict = {'mu': mu, 'sig': sig, 'prob': prob, 'pi': pi}
        return param_dict

    def forward(self, data_dict, loss=False):
        image = data_dict['img']
        batch_size = image.shape[0]
        net_data_dict = self.__sync_batch_and_device__(batch_size, image.device.index)

        fmaps = self.net['ft_extractor'].forward(image)
        out_tensors = self.__get_output_tensors__(fmaps, net_data_dict, batch_size)
        output_dict = self.__get_mixture_params__(out_tensors, net_data_dict)

        if loss:
            loss_dict, value_dict = self.loss_func.forward(output_dict, data_dict)
            return output_dict, loss_dict, value_dict
        else:
            return output_dict

