import os
import math
import torch
import numpy as np
from torchvision.ops import box_iou

epsilon = 1e-12
float_epsilon = 1e-36
double_epsilon = 1e-300


# def tuple2dict(tuple_data):
#     from atss_core.structures.image_list import to_image_list
#     img = to_image_list(tuple_data[0])
#     img = img.tensors
#     labels = torch.stack(tuple_data[1], dim=0)
#     boxes, n_boxes = torch.stack(tuple_data[2], dim=0), torch.stack(tuple_data[3], dim=0)
#     img_size = torch.stack(tuple_data[4], dim=0)
#     data_dict = {
#         'img': img.cuda(), 'labels': labels.cuda(), 'boxes': boxes.cuda(),
#         'n_boxes': n_boxes.cuda(), 'img_size': img_size.cuda()
#     }
#     return data_dict


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def cvt_torch2numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = tensor.detach().numpy()
    elif isinstance(tensor, list) or isinstance(tensor, tuple):
        for i in range(len(tensor)):
            tensor[i] = cvt_torch2numpy(tensor[i])
    elif isinstance(tensor, dict):
        for key in tensor.keys():
            tensor[key] = cvt_torch2numpy(tensor[key])
    return tensor


def print_matrix(matrix, form='%.1f'):
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            print(form % matrix[r, c], end=' ')
        print('')


def cvt_int2onehot(integer, onehot_size):
    if len(integer.shape) > 2:
        integer = integer.squeeze(dim=2)
    onehot = torch.zeros(integer.shape + (onehot_size,)).float().cuda()
    dim = len(onehot.shape) - 1
    onehot.scatter_(dim, integer.unsqueeze(dim=dim), 1)
    return onehot


def log_sum_exp(x, dim):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), dim=dim, keepdim=True)) + x_max


def clip_boxes(boxes, size):
    boxes[:, :, [0, 2]] = torch.clamp(boxes[:, :, [0, 2]], min=0, max=size[1] - 1)
    boxes[:, :, [1, 3]] = torch.clamp(boxes[:, :, [1, 3]], min=0, max=size[0] - 1)
    return boxes


def clip_boxes_s(boxes_s, size, numpy=False):
    if numpy:
        boxes_s[:, [0, 2]] = np.clip(boxes_s[:, [0, 2]], a_min=0, a_max=size[1] - 1)
        boxes_s[:, [1, 3]] = np.clip(boxes_s[:, [1, 3]], a_min=0, a_max=size[0] - 1)
    else:
        boxes_s[:, [0, 2]] = torch.clamp(boxes_s[:, [0, 2]], min=0, max=size[1] - 1)
        boxes_s[:, [1, 3]] = torch.clamp(boxes_s[:, [1, 3]], min=0, max=size[0] - 1)
    return boxes_s


def sort_boxes_s(boxes_s, confs_s, labels_s=None):
    sorted_confs_s, sorted_idxs = torch.sort(confs_s, dim=0, descending=True)
    if len(sorted_idxs.shape) == 2:
        sorted_idxs = torch.squeeze(sorted_idxs, dim=1)
    sorted_boxes_s = boxes_s[sorted_idxs]

    if labels_s is None:
        return sorted_boxes_s, sorted_confs_s
    else:
        sorted_labels_s = labels_s[sorted_idxs]
        return sorted_boxes_s, sorted_confs_s, sorted_labels_s


def sort_boxes_s_np(boxes_s, confs_s, labels_s=None):
    sorted_idxs = np.argsort(confs_s, axis=0)[::-1]
    if len(sorted_idxs.shape) == 2:
        sorted_idxs = np.squeeze(sorted_idxs, axis=1)
    sorted_boxes_s = boxes_s[sorted_idxs]
    sorted_confs_s = confs_s[sorted_idxs]

    if labels_s is None:
        return sorted_boxes_s, sorted_confs_s
    else:
        sorted_labels_s = labels_s[sorted_idxs]
        return sorted_boxes_s, sorted_confs_s, sorted_labels_s


def calc_jaccard_torch(boxes_a, boxes_b):
    # from https://github.com/luuuyi/RefineDet.PyTorch

    _boxes_a = boxes_a.clone()
    _boxes_b = boxes_b.clone()

    def intersect_torch(__boxes_a, __boxes_b):
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          __boxes_a: (tensor) bounding boxes, Shape: [A,4]. 여기서 A는 truths의 bounding box
          __boxes_b: (tensor) bounding boxes, Shape: [B,4]. 여기서 B는 anchor들의 bounding box
        Return:
          (tensor) intersection area, Shape: [A,B].
        """
        A = __boxes_a.size(0)
        B = __boxes_b.size(0)
        max_xy = torch.min(__boxes_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           __boxes_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(__boxes_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           __boxes_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from anchorbox layers, Shape: [num_anchors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect_torch(_boxes_a, _boxes_b)
    area_a = ((_boxes_a[:, 2] - _boxes_a[:, 0]) *
              (_boxes_a[:, 3] - _boxes_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((_boxes_b[:, 2] - _boxes_b[:, 0]) *
              (_boxes_b[:, 3] - _boxes_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def calc_jaccard_numpy(boxes_a, box_b):
    # From: https://github.com/amdegroot/ssd.pytorch
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        boxes_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    def intersect_numpy(_boxes_a, _boxes_b):
        max_xy = np.minimum(_boxes_a[:, 2:4], _boxes_b[2:4])
        min_xy = np.maximum(_boxes_a[:, 0:2], _boxes_b[0:2])
        _inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
        return _inter[:, 0] * _inter[:, 1]

    inter = intersect_numpy(boxes_a, box_b)
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) *
              (boxes_a[:, 3] - boxes_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def gaussian_pdf(x, mu, sig):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    # print(x.shape, mu.shape, sig.shape)
    dist = ((x - mu) / sig) ** 2
    # result = (x - mu) / sig

    result = -0.5 * dist
    result = torch.exp(result) / (sig * math.sqrt(2.0 * math.pi))
    return result


def cauchy_pdf(x, loc, sc):
    # x:    (batch, #x, 4, 1)
    # loc:  (batch, 1, 4, #comp)
    # sc:   (batch, 1, 4, #comp)

    dist = ((x - loc) / sc) ** 2
    # dist: (batch, #x, 4, #comp)
    result = 1 / (math.pi * sc * (dist + 1))
    return result


def mm_pdf(mu, sig, pi, points, sum_comp=True, comp_pdf=gaussian_pdf):

    mu, sig, pi = mu.unsqueeze(dim=1), sig.unsqueeze(dim=1), pi.unsqueeze(dim=1)
    points = points.unsqueeze(dim=3)
    # mu, sig shape:    (batch, 1, 4, n_gauss), xywh
    # pi shape:         (batch, 1, 1, n_gauss)
    # points shape:     (batch, n_points, 4, 1), xywh

    result = comp_pdf(points, mu, sig)
    # result.shape:     (batch, n_points, 4, n_gauss)
    result = torch.prod(result, dim=2, keepdim=True)
    # result.shape:     (batch, n_points, 1, n_gauss)

    if pi is not None:
        result = pi * result
        # result.shape: (batch, n_points, 1, n_gauss)
    if sum_comp:
        result = torch.sum(result, dim=3)
        # result.shape: (batch, n_points, 1)
    return result


def mm_pdf_s(mu_s, sig_s, pi_s, points_s, sum_comp=True, comp_pdf=gaussian_pdf):
    mu, sig, pi = mu_s.unsqueeze(dim=0), sig_s.unsqueeze(dim=0), pi_s.unsqueeze(dim=0)
    points_s = points_s.unsqueeze(dim=2)
    # mu, sig shape:    (1, 4, n_gauss), xywh
    # pi shape:         (1, 1, n_gauss)
    # points shape:     (n_points, 4, 1), xywh

    result = comp_pdf(points_s, mu_s, sig_s)
    # result.shape:     (n_points, 4, n_gauss)
    result = torch.prod(result, dim=1, keepdim=True)
    # result.shape:     (n_points, 1, n_gauss)
    if pi is not None:
        result = (pi * result)[:, 0]
        # result.shape: (n_points, n_gauss)
    if sum_comp:
        result = torch.sum(result, dim=1)
        # result.shape: (n_points)
    return result


def category_pmf(clsprobs, onehot_labels):
    # clsprobs: (batch, #classes, #comp)
    # labels:   (batch, #samples, #classes)
    clsprobs = clsprobs.unsqueeze(dim=1)
    onehot_labels = onehot_labels.unsqueeze(dim=3)
    cat_probs = torch.prod(clsprobs ** onehot_labels, dim=2, keepdim=True)
    return cat_probs


def category_pmf_s(clsprobs, onehot_labels):
    # clsprobs: (#classes, #comp)
    # labels:   (#samples, #classes)
    clsprobs = clsprobs.unsqueeze(dim=0)
    onehot_labels = onehot_labels.unsqueeze(dim=2)
    cat_probs = torch.prod(clsprobs ** onehot_labels, dim=1)
    return cat_probs


def bernoulli_pmf_s(probs, labels):
    probs = probs.unsqueeze(dim=0)
    labels = labels.unsqueeze(dim=2)
    # probs:    (1, #classes, #comp)
    # labels:   (#samples, #classes, 1)
    ber_probs = (labels * probs) + ((1 - labels) * (1 - probs))
    ber_probs = torch.prod(ber_probs, dim=1)
    return ber_probs


def sample_coords_from_mog(mu, sig, pi, n_samples, sampling_noise=True):
    # mu.shape: (batch_size, 4, #comp)
    # sig.shape: (batch_size, 4, #comp)
    # pi.shape: (batch_size, 1, #comp)
    # print(mu.shape, sig.shape, pi.shape, n_samples)
    _mu = cvt_torch2numpy(mu)
    _sig = cvt_torch2numpy(sig)
    _pi = cvt_torch2numpy(pi)
    n_gauss = _pi[0].shape[1]

    gen_coords = list()
    for mu_s, sig_s, pi_s in zip(_mu, _sig, _pi):
        pi_s = pi_s[0]
        gauss_nums = np.random.choice(n_gauss, size=n_samples, p=pi_s)
        normal_noises = np.random.randn(n_samples * 4).reshape((4, n_samples)) if sampling_noise else 0
        gen_coords_s = mu_s[:, gauss_nums] + normal_noises * sig_s[:, gauss_nums]
        gen_coords.append(np.expand_dims(gen_coords_s, axis=0))

    gen_coords = np.concatenate(gen_coords, axis=0)
    gen_coords = torch.from_numpy(gen_coords).float().cuda()
    gen_coords = gen_coords.transpose(1, 2)
    return gen_coords


def create_coord_map(coord_map_size, coord_range):
    # gauss_w: 4 --> ((0, 1, 2, 3), ...)
    x_map = np.array(list(range(coord_map_size[1])) * coord_map_size[0]).astype(np.float32)
    y_map = np.array(list(range(coord_map_size[0])) * coord_map_size[1]).astype(np.float32)

    x_map = x_map.reshape((1, 1, coord_map_size[0], coord_map_size[1]))
    y_map = y_map.reshape((1, 1, coord_map_size[1], coord_map_size[0]))
    y_map = y_map.transpose((0, 1, 3, 2))

    # coord_w: 100 --> unit_intv_w: 25
    unit_intv_w = coord_range[1] / coord_map_size[1]
    unit_intv_h = coord_range[0] / coord_map_size[0]

    # (0, 1, 2, 3) * 25 + 12.5 == (12.5, 37.5, 62.5, 87.5)
    x_map = x_map * unit_intv_w + unit_intv_w / 2
    y_map = y_map * unit_intv_h + unit_intv_h / 2
    return np.concatenate((x_map, y_map), axis=1)


def sample_boxes_s(pi_s, mu_s, n_samples):
    # pi_s: (1, #comp)
    # mu_s: (4, #comp)
    sampler = torch.distributions.categorical.Categorical(pi_s[0])
    sampled_indices = sampler.sample((n_samples,))
    sampled_boxes_s = mu_s[:, sampled_indices]
    return sampled_boxes_s
