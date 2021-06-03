import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from lib import util as lib_util


def get_parent_module(root_module, name):
    keys = name.split('.')[:-1]
    cur_module = root_module
    for key in keys:
        cur_module = cur_module.__getattr__(key)
    return cur_module


def init_modules(module_list, init_method='xavier_uniform'):
    if (init_method is None) or (init_method == 'None') or (init_method == 'none'):
        pass
    else:
        init_dict = {
            'xavier_uniform': init_modules_xavier_uniform,
            'xavier_normal': init_modules_xavier_normal,
            'kaiming_uniform': init_modules_kaiming_uniform,
            'kaiming_normal': init_modules_kaiming_normal,
            'uniform': init_modules_uniform,
            'normal': init_modules_normal,
            'othogonal': init_modules_othogonal,
            'constant': init_modules_constant,
        }
        init_dict[init_method](module_list)


def init_modules_xavier_uniform(module_list):
    for m in module_list:
        if isinstance(m, nn.Conv2d) or \
                isinstance(m, nn.ConvTranspose2d) or \
                isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Sequential):
            init_modules_xavier_uniform(m)
        else:
            pass
            # raise NameError('ModuleInitError: %s' % str(type(m)))


def init_modules_xavier_normal(module_list):
    for m in module_list:
        if isinstance(m, nn.Conv2d) or \
                isinstance(m, nn.ConvTranspose2d) or \
                isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Sequential):
            init_modules_xavier_normal(m)
        else:
            pass


def init_modules_kaiming_uniform(module_list):
    # Conv default a=0.0
    for m in module_list:
        if isinstance(m, nn.Conv2d) or \
                isinstance(m, nn.ConvTranspose2d) or \
                isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Sequential):
            init_modules_kaiming_uniform(m)
        else:
            pass


def init_modules_kaiming_normal(module_list):
    for m in module_list:
        if isinstance(m, nn.Conv2d) or \
                isinstance(m, nn.ConvTranspose2d) or \
                isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Sequential):
            init_modules_kaiming_normal(m)
        else:
            pass


def init_modules_uniform(module_list):
    for m in module_list:
        if isinstance(m, nn.Conv2d) or \
                isinstance(m, nn.ConvTranspose2d) or \
                isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.uniform_(m.weight, a=-0.1, b=0.1)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Sequential):
            init_modules_uniform(m)
        else:
            pass


def init_modules_normal(module_list):
    for m in module_list:
        if isinstance(m, nn.Conv2d) or \
                isinstance(m, nn.ConvTranspose2d) or \
                isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.normal_(m.weight, mean=0, std=0.1)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Sequential):
            init_modules_normal(m)
        else:
            pass


def init_modules_othogonal(module_list):
    for m in module_list:
        if isinstance(m, nn.Conv2d) or \
                isinstance(m, nn.ConvTranspose2d) or \
                isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Sequential):
            init_modules_othogonal(m)
        else:
            pass


def init_modules_constant(module_list):
    for m in module_list:
        if isinstance(m, nn.Conv2d) or \
                isinstance(m, nn.ConvTranspose2d) or \
                isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 0.001)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Sequential):
            init_modules_constant(m)
        else:
            pass


def __cvt_xywh2ltrb__(_o1):
    _o1[:, 0] = _o1[:, 0] - (_o1[:, 2] / 2)
    _o1[:, 1] = _o1[:, 1] - (_o1[:, 3] / 2)
    _o1[:, 2] = _o1[:, 0] + _o1[:, 2]
    _o1[:, 3] = _o1[:, 1] + _o1[:, 3]
    return _o1


def create_xy_maps(batch_size, fmap_sizes, coord_range):
    xy_maps = list()
    for fmap_size in fmap_sizes:
        xy_map = lib_util.create_coord_map(fmap_size, coord_range)
        xy_maps.append(torch.from_numpy(xy_map).float().requires_grad_(False))
    xy_maps = [torch.cat([xy_map] * batch_size, dim=0) for xy_map in xy_maps]
    return xy_maps


def create_def_coord(batch_size, output_sizes, coord_range, with_wh=False):
    coord_size = 4 if with_wh else 2
    def_coords = list()
    num_def_coords = len(output_sizes)
    for lv, output_size in enumerate(output_sizes):
        def_coord = create_box_coord_map(output_size, num_def_coords, coord_range, with_wh=with_wh)
        if with_wh:
            def_coord = def_coord[lv]
        def_coords.append(torch.from_numpy(def_coord).float().view(1, coord_size, -1).requires_grad_(False))
    def_coord = torch.cat([torch.cat(def_coords, dim=2)] * batch_size, dim=0)
    return def_coord

def create_box_coord_map(output_size, output_ch, coord_range, with_wh=False):
    box_coord_map = lib_util.create_coord_map(output_size, coord_range)

    if with_wh:
        box_coord_map = np.zeros((output_ch, 4, output_size[0], output_size[1])).astype(np.float32)
        box_coord_map[:, :2] += lib_util.create_coord_map(output_size, coord_range)

        # gauss_ch: 4 --> ((0, 1, 2, 3), ...)
        ch_map = np.array(list(range(output_ch)))

        # coord_w: 100 --> unit_intv_w: 20 = 100 / (4 + 1)
        unit_intv_w = coord_range[1] / (output_ch + 1.0)
        unit_intv_h = coord_range[0] / (output_ch + 1.0)

        # ((0, 1, 2, 3) + 1) * 20 == (20, 40, 60, 80)
        w_map = (ch_map + 1) * unit_intv_w
        h_map = (ch_map + 1) * unit_intv_h

        # ((20, 40, 60, 80) / 100)^2 == (0.04, 0.16, 0.36, 0.64)
        # (0.04, 0.16, 0.36, 0.64) * 100 == (4, 16, 36, 64)
        w_map = ((w_map / coord_range[1]) ** 2) * coord_range[1]
        h_map = ((h_map / coord_range[0]) ** 2) * coord_range[0]

        w_map = w_map.reshape((output_ch, 1, 1))
        h_map = h_map.reshape((output_ch, 1, 1))
        box_coord_map[:, 2] = w_map
        box_coord_map[:, 3] = h_map

    # print(box_coord_map.shape)
    # box_coord_map = np.transpose(box_coord_map, axes=(1, 0, 2, 3))
    # box_coord_map = np.expand_dims(box_coord_map, axis=0)
    # (1, 4, gauss_ch, gauss_h, gauss_w)
    # print(box_coord_map.shape)
    # exit()
    return box_coord_map


def create_limit_scale(batch_size, output_sizes, coord_range, limit_factor):
    # n_lv_mix_comps = [output_size[0] * output_size[1] for output_size in output_sizes]

    lv_x_limit_scales, lv_y_limit_scales = list(), list()
    # for i, n_lv_mix_comp in enumerate(n_lv_mix_comps):
    for output_size in output_sizes:
        x_limit_scale = (coord_range[1] / output_size[1]) * limit_factor
        y_limit_scale = (coord_range[0] / output_size[0]) * limit_factor

        n_lv_mix_comp = output_size[0] * output_size[1]
        lv_x_limit_scales.append(x_limit_scale * torch.ones((1, 1, n_lv_mix_comp)).float())
        lv_y_limit_scales.append(y_limit_scale * torch.ones((1, 1, n_lv_mix_comp)).float())

    x_limit_scale = torch.cat(lv_x_limit_scales, dim=2)
    y_limit_scale = torch.cat(lv_y_limit_scales, dim=2)
    limit_scale = torch.cat([x_limit_scale, y_limit_scale], dim=1).requires_grad_(False)
    limit_scale = torch.cat([limit_scale] * batch_size, dim=0)
    return limit_scale


def create_id_basis_vector(id_basis_vector, id_bit_size):
    id_basis_vector1 = id_basis_vector + [1.0]
    id_basis_vector2 = id_basis_vector + [0.0]

    if id_bit_size > 1:
        return1 = create_id_basis_vector(id_basis_vector1, id_bit_size - 1)
        return2 = create_id_basis_vector(id_basis_vector2, id_bit_size - 1)
        return return1 + return2
    else:
        return [id_basis_vector1, id_basis_vector2]


def create_id_basis(batch_size, n_classes, id_bit_size, output_sizes):
    n_comp = sum([out_h * out_w for out_h, out_w in output_sizes])

    id_basis_vector_list = create_id_basis_vector([], id_bit_size)
    id_basis = np.array(id_basis_vector_list).transpose((1, 0))
    id_basis = id_basis.reshape((1, 1, id_basis.shape[0], id_basis.shape[1], 1))
    zero_arr = np.zeros((batch_size, n_classes, id_basis.shape[0], id_basis.shape[1], n_comp))
    id_basis = torch.from_numpy(id_basis + zero_arr).float()
    return id_basis


def create_def_id_bit(batch_size, n_classes, id_bit_size, output_sizes, sigmoid_factor=2):
    assert id_bit_size % 2 == 0

    id_basis_vector_list = create_id_basis_vector([], int(id_bit_size / 2))
    id_basis = torch.from_numpy(np.array(id_basis_vector_list).transpose((1, 0))).float()
    id_basis_map1 = torch.cat([id_basis.unsqueeze(2)] * id_basis.shape[1], dim=2)
    id_basis_map2 = id_basis_map1.transpose(1, 2)
    id_basis_map = torch.cat([id_basis_map1, id_basis_map2], dim=0).unsqueeze(0)
    # id_basis.shape:       (half_id_bit_size, 2^half_id_bit_size, 1)
    # id_basis_map1.shape:  (half_id_bit_size, 2^half_id_bit_size, 2^half_id_bit_size)
    # id_basis_map2.shape:  (half_id_bit_size, 2^half_id_bit_size, 2^half_id_bit_size)
    # id_basis_map.shape:   (1, id_bit_size, 2^half_id_bit_size, 2^half_id_bit_size)

    def_id_bit = list()
    for output_size in output_sizes:
        def_id_bit_map = func.interpolate(
            id_basis_map, size=output_size, mode='bilinear', align_corners=False)
        def_id_bit.append(def_id_bit_map.view(id_bit_size, -1))
        # def_id_bit_map.shape: (id_bit_size, out_h, out_w)

    def_id_bit = torch.cat(def_id_bit, dim=1)
    def_id_bit = (def_id_bit * 2 * sigmoid_factor) - sigmoid_factor
    def_id_bit = def_id_bit.view(1, 1, def_id_bit.shape[0], def_id_bit.shape[1])
    def_id_bit = torch.cat([torch.cat([def_id_bit] * batch_size, dim=0)] * n_classes, dim=1)
    # def_id_bit.shape: (batch_size, n_class, id_bit_size, n_comp)
    return def_id_bit


def create_onehot_map(batch_size, map_size, onehot_size):
    onehot_map = torch.zeros((map_size[0], map_size[1], onehot_size * 2))
    for i in range(onehot_size):
        onehot_map[i::3, :, i] = 1.0
        onehot_map[:, i::3, onehot_size + i] = 1.0
    onehot_map = onehot_map.permute((2, 0, 1))
    onehot_map = torch.stack([onehot_map] * batch_size, dim=0)
    return onehot_map
