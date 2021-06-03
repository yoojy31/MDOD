import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision.ops import box_iou
from lib.network import __util__ as net_util
from lib import util as lib_util


def get_actv_func(actv_func_key):
    return {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.2, inplace=True),
        'swish': Swish(),
        'mem_eff_swish': MemoryEfficientSwish(),
    }[actv_func_key]


class Level5FPN(nn.Module):
    def __init__(self, input_chs, num_filters, upsampling, actv_func):
        super(Level5FPN, self).__init__()
        self.stage_c6 = nn.Conv2d(input_chs[2], num_filters, 3, 2, 1)
        self.stage_c7 = nn.Sequential(
            actv_func, nn.Conv2d(num_filters, num_filters, 3, 2, 1))

        self.stage_p5_1 = nn.Conv2d(input_chs[2], num_filters, 1, 1, 0)
        self.stage_p5_2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.stage_p5_up = nn.Upsample(scale_factor=2, mode=upsampling)

        self.stage_p4_1 = nn.Conv2d(input_chs[1], num_filters, 1, 1, 0)
        self.stage_p4_2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.stage_p4_up = nn.Upsample(scale_factor=2, mode=upsampling)

        self.stage_p3_1 = nn.Conv2d(input_chs[0], num_filters, 1, 1, 0)
        self.stage_p3_2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)

    def forward(self, fmaps):
        # fmap_p6 = self.net['stage_c6'].forward(fmap_c5)
        fmap_p6 = self.stage_c6.forward(fmaps[2])
        fmap_p7 = self.stage_c7.forward(fmap_p6)

        # _fmap_p5 = self.net['stage_p5_1'].forward(fmap_c5)
        _fmap_p5 = self.stage_p5_1.forward(fmaps[2])
        fmap_p5 = self.stage_p5_2.forward(_fmap_p5)
        _fmap_p5_up = self.stage_p5_up.forward(_fmap_p5)

        # _fmap_p4 = self.net['stage_p4_1'].forward(fmap_c4) + _fmap_p5_up
        _fmap_p4 = self.stage_p4_1.forward(fmaps[1]) + _fmap_p5_up
        fmap_p4 = self.stage_p4_2.forward(_fmap_p4)
        _fmap_p4_up = self.stage_p4_up.forward(_fmap_p4)

        # _fmap_p3 = self.net['stage_p3_1'].forward(fmap_c3) + _fmap_p4_up
        _fmap_p3 = self.stage_p3_1.forward(fmaps[0]) + _fmap_p4_up
        fmap_p3 = self.stage_p3_2.forward(_fmap_p3)
        return [fmap_p3, fmap_p4, fmap_p5, fmap_p6, fmap_p7]


class VarLevel5FPN(nn.Module):
    def __init__(self, input_chs, num_filters, upsampling, actv_func):
        super(VarLevel5FPN, self).__init__()
        self.up_mode = upsampling

        self.stage_c6 = nn.Conv2d(input_chs[2], num_filters, 3, 2, 1)
        self.stage_c7 = nn.Sequential(
            actv_func, nn.Conv2d(num_filters, num_filters, 3, 2, 1))

        self.stage_p5_1 = nn.Conv2d(input_chs[2], num_filters, 1, 1, 0)
        self.stage_p5_2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        # self.stage_p5_up = nn.Upsample(scale_factor=2, mode=upsampling)

        self.stage_p4_1 = nn.Conv2d(input_chs[1], num_filters, 1, 1, 0)
        self.stage_p4_2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        # self.stage_p4_up = nn.Upsample(scale_factor=2, mode=upsampling)

        self.stage_p3_1 = nn.Conv2d(input_chs[0], num_filters, 1, 1, 0)
        self.stage_p3_2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)

    def forward(self, fmaps):
        fmap_p6 = self.stage_c6.forward(fmaps[2])
        fmap_p7 = self.stage_c7.forward(fmap_p6)

        _fmap_p5 = self.stage_p5_1.forward(fmaps[2])
        fmap_p5 = self.stage_p5_2.forward(_fmap_p5)
        # _fmap_p5_up = self.stage_p5_up.forward(_fmap_p5)
        _fmap_p5_up = func.upsample(_fmap_p5, size=fmaps[1].shape[2:], mode=self.up_mode)

        _fmap_p4 = self.stage_p4_1.forward(fmaps[1]) + _fmap_p5_up
        fmap_p4 = self.stage_p4_2.forward(_fmap_p4)
        # _fmap_p4_up = self.stage_p4_up.forward(_fmap_p4)
        _fmap_p4_up = func.upsample(_fmap_p4, size=fmaps[0].shape[2:], mode=self.up_mode)

        _fmap_p3 = self.stage_p3_1.forward(fmaps[0]) + _fmap_p4_up
        fmap_p3 = self.stage_p3_2.forward(_fmap_p3)
        return [fmap_p3, fmap_p4, fmap_p5, fmap_p6, fmap_p7]


class MyLevel5FPN(nn.Module):
    def __init__(self, input_chs, num_filters, upsampling):
        super(MyLevel5FPN, self).__init__()
        self.stage_c6 = nn.Conv2d(input_chs[3], num_filters, 3, 2, 1)

        self.stage_p5_1 = nn.Conv2d(input_chs[3], num_filters, 1, 1, 0)
        self.stage_p5_2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.stage_p5_up = nn.Upsample(scale_factor=2, mode=upsampling)

        self.stage_p4_1 = nn.Conv2d(input_chs[2], num_filters, 1, 1, 0)
        self.stage_p4_2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.stage_p4_up = nn.Upsample(scale_factor=2, mode=upsampling)

        self.stage_p3_1 = nn.Conv2d(input_chs[1], num_filters, 1, 1, 0)
        self.stage_p3_2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.stage_p3_up = nn.Upsample(scale_factor=2, mode=upsampling)

        self.stage_p2_1 = nn.Conv2d(input_chs[0], num_filters, 1, 1, 0)
        self.stage_p2_2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)

    def forward(self, fmaps):
        fmap_p6 = self.stage_c6.forward(fmaps[3])

        _fmap_p5 = self.stage_p5_1.forward(fmaps[3])
        fmap_p5 = self.stage_p5_2.forward(_fmap_p5)
        _fmap_p5_up = self.stage_p5_up.forward(_fmap_p5)

        _fmap_p4 = self.stage_p4_1.forward(fmaps[2]) + _fmap_p5_up
        fmap_p4 = self.stage_p4_2.forward(_fmap_p4)
        _fmap_p4_up = self.stage_p4_up.forward(_fmap_p4)

        _fmap_p3 = self.stage_p3_1.forward(fmaps[1]) + _fmap_p4_up
        fmap_p3 = self.stage_p3_2.forward(_fmap_p3)
        _fmap_p3_up = self.stage_p4_up.forward(_fmap_p3)

        _fmap_p2 = self.stage_p2_1.forward(fmaps[0]) + _fmap_p3_up
        fmap_p2 = self.stage_p2_2.forward(_fmap_p2)
        return [fmap_p2, fmap_p3, fmap_p4, fmap_p5, fmap_p6]


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class StepDownConv(nn.Module):
    def __init__(self, n_step, in_ch, out_ch, kernel_size, padding, bias=True, mode='bilinear'):
        super(StepDownConv, self).__init__()
        layers = list()
        for i in range(n_step):
            scale_factor = 0.5 ** ((i + 1) / n_step)
            step_in_ch = (out_ch / in_ch) ** (i / n_step)
            step_out_ch = (out_ch / in_ch) ** ((i + 1) / n_step)
            layers.append(nn.Conv2d(step_in_ch, step_out_ch, kernel_size, 1, padding, bias))
            layers.append(nn.Upsample(scale_factor=scale_factor, mode=mode))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers.forward(x)


class DownPixelShuffle(nn.Module):
    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 2 * 2, h // 2, w // 2)
        return x


class DynamicRouting(nn.Module):
    def __init__(self, n_routing):
        super(DynamicRouting, self).__init__()
        self.n_routing = n_routing

    @ staticmethod
    def squash(s, dim=1):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s ** 2, dim=dim, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x, coef=None, y=None):
        assert not ((x is None) and (y is None))
        # x:    (batch, capsule_ch, n_in_capsule)
        # coef: (batch, n_out_capsule, n_in_capsule)
        # y:    (batch, capsule_ch, n_out_capsule)

        coef = 0 if coef is None else coef
        for i in range(self.n_routing):
            if y is not None:
                v = self.squash(y).transpose(1, 2)
                coef += torch.bmm(v, x)
                # v:    (batch, n_out_capsule, capsule_ch)
                # coef: (batch, n_out_capsule, n_in_capsule)

            nor_coef = torch.softmax(coef, dim=2)
            # nor_coef = torch.softmax(coef, dim=1)
            y = torch.sum(nor_coef.unsqueeze(dim=1) *
                          x.unsqueeze(dim=2), dim=3)
            # y:    (batch, capsule_ch, n_out_capsule)
        return y, coef


class MixtureModule(nn.Module):
    def __init__(self, mixture_dim, coord_range, max_mixture_map_size,
                 max_batch_size=1, std_factor=0.05):
        super(MixtureModule, self).__init__()

        self.split_indices = [1, mixture_dim, mixture_dim]
        self.input_ch = sum(self.split_indices)
        self.std_factor = std_factor
        self.center_offset = net_util.create_xy_maps(
            max_batch_size, [max_mixture_map_size], coord_range)[0]

    def sync_batch_and_device(self, batch_size, device_idx):
        if device_idx != self.center_offset.device.index:
            self.center_offset = self.center_offset.cuda()
        center_offset = self.center_offset[:batch_size]
        return center_offset

    def forward(self, x):
        batch_size, device_idx = x.shape[0], x.device.index
        center_offset = self.sync_batch_and_device(batch_size, device_idx)

        x[:, 1:3] += center_offset
        x = x.view(batch_size, self.input_ch, -1)
        wh = torch.cat([x[:, 3:5]] * 2, dim=1)

        o1, o2, o3 = torch.split(x, self.split_indices, dim=1)
        # pi = torch.softmax(o1, dim=2)
        pi = torch.sigmoid(o1) / torch.sum(torch.sigmoid(o1), dim=2, keepdim=True)
        mu = net_util.__cvt_xywh2ltrb__(o2)
        sig = torch.max(func.softplus(o3), wh * self.std_factor)
        return pi, mu, sig


class MixtureRouting(nn.Module):
    def __init__(self, coef_detach=True):
        super(MixtureRouting, self).__init__()
        self.coef_detach = coef_detach

    def forward(self, x, pi, mu, sig):
        points = mu.transpose(1, 2)
        lh = lib_util.mm_pdf(
            mu, sig, pi, points, sum_comp=False,
            comp_pdf=lib_util.cauchy_pdf)[:, :, 0]
        coef = lh / torch.sum(lh, dim=2, keepdim=True)

        coef = coef.detach() if self.coef_detach else coef
        y = torch.bmm(x, coef)
        # x:    (batch, ft_dim, #comp)
        # coeff:  (batch, #comp, #comp)
        # y:    (batch, gt_dim, #comp)
        return y


class MixtureAttention(nn.Module):
    def __init__(self, coef_detach=True):
        super(MixtureAttention, self).__init__()
        self.coef_detach = coef_detach

    def forward(self, x, pi, mu, sig):
        points = mu.transpose(1, 2)
        lh = lib_util.mm_pdf(
            mu, sig, pi, points, sum_comp=False,
            comp_pdf=lib_util.cauchy_pdf)[:, :, 0]
        _coef = lh / torch.sum(lh, dim=2, keepdim=True)
        coef = torch.diagonal(_coef, dim1=-2, dim2=-1).unsqueeze(dim=1)

        coef = coef.detach() if self.coef_detach else coef
        y = torch.bmm(x, coef)
        # y = coef * x
        # x:    (batch, ft_dim, #comp)
        # coeff:  (batch, 1, #comp)
        # y:    (batch, gt_dim, #comp)
        return y


class MixtureAggregation(nn.Module):
    def __init__(self, coef_detach=True):
        super(MixtureAggregation, self).__init__()
        self.coef_detach = coef_detach

    def forward(self, x, pi, mu, sig):
        points = mu.transpose(1, 2)
        lh = lib_util.mm_pdf(
            mu, sig, pi, points, sum_comp=False,
            comp_pdf=lib_util.cauchy_pdf)[:, :, 0]
        coef = lh / torch.sum(lh, dim=1, keepdim=True)

        coef = coef.detach() if self.coef_detach else coef
        y = torch.bmm(x, coef)
        # x:    (batch, ft_dim, #comp)
        # coef: (batch, #comp, #comp)
        # y:    (batch, gt_dim, #comp)
        return y


class BoxTopKPool(nn.Module):
    def __init__(self, top_k):
        super(BoxTopKPool, self).__init__()
        self.top_k = top_k

    def forward(self, pool_score):
        pool_score, indices = torch.topk(pool_score, k=self.top_k, dim=2)
        # pool_score = pool_score.unsqueeze(dim=1)
        return pool_score, indices


class BoxPool(nn.Module):
    def __init__(self, threshold=0.7, max_boxes=2134):
        super(BoxPool, self).__init__()
        self.threshold = threshold
        box_indices = torch.from_numpy(
            np.array(list(range(max_boxes))))
        self.box_indices = box_indices.unsqueeze(dim=0).cuda()

    @staticmethod
    def thresholding(x, threshold):
        x.masked_fill_(x >= threshold, 1.0)
        x.masked_fill_(x < threshold, 0.0)
        return x

    def forward(self, box, score):
        batch_size = box.shape[0]
        n_boxes = box.shape[2]
        n_classes = score.shape[1]
        box_indices = self.box_indices[:, :n_boxes]
        score = score.transpose(dim0=1, dim1=2)
        # feature:  (batch, ft_size, #box)
        # boxes:    (batch, 4, #box)
        # score:    (batch, #box, #class)

        box = box.transpose(dim0=1, dim1=2)
        iou = torch.stack([box_iou(boxes_s, boxes_s) for boxes_s in box], dim=0)
        iou_mask = self.thresholding(iou, self.threshold)
        # iou:          (batch, #box, #box)
        # iou_mask:     (batch, #box, #box)

        if n_classes > 1:
            pool_mask = list()
            pool_mask.append(torch.ones(batch_size, n_boxes).cuda())
            for i in range(1, n_classes):
                score_c = score[:, :, i:i + 1]

                adjc_score_c = iou_mask * score_c
                max_idx_c = torch.argmax(adjc_score_c, dim=1)
                # adjc_score_c:     (batch, #box, #box)
                # max_idx_c:        (barch, #box)

                pool_mask_c = (max_idx_c == box_indices).long()
                pool_mask.append(pool_mask_c)
                # pool_mask_c:      (batch, #box)

            pool_mask = torch.stack(pool_mask, dim=1)
            # pool_mask:    (batch, #class, #box)
        else:
            adjc_score = iou_mask * score
            max_idx = torch.argmax(adjc_score, dim=1)
            # adjc_score:       (batch, #box, #box)
            # max_idx:          (barch, #box)

            pool_mask = (max_idx == box_indices).long()
            # pool_mask:    (batch, 1, #box)
        return pool_mask


class WeightedBoxPool(nn.Module):
    def __init__(self, thresholds=None, max_boxes=2134):
        super(WeightedBoxPool, self).__init__()
        if thresholds is None:
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

        self.n_threshold = len(thresholds)
        thresholds = torch.from_numpy(np.array(thresholds)).cuda()
        self.thresholds = thresholds.view(1, 1, 1, self.n_threshold)
        self.max_boxes = max_boxes

        box_indices = torch.from_numpy(np.array(list(range(max_boxes))))
        self.box_indices = box_indices.unsqueeze(dim=0).unsqueeze(dim=2).cuda()

    def thresholding(self, x):
        x.masked_fill_(x >= self.thresholds, 1.0)
        x.masked_fill_(x < self.thresholds, 0.0)
        return x

    def forward(self, mask_weight, box, score=None):
        n_boxes = box.shape[2]
        # feature:  (batch, ft_size, #box)
        # boxes:    (batch, 4, #box)
        # score:    (batch, 1, #box)

        box = box.transpose(dim0=1, dim1=2)
        if box.shape[0] == 1:
            iou_pair = box_iou(box[0], box[0]).unsqueeze(0)
        else:
            iou_pair = torch.stack([box_iou(boxes_s, boxes_s) for boxes_s in box], dim=0)
        # iou_pair:     (batch, #box, #box)

        iou_pair = torch.stack([iou_pair] * self.n_threshold, dim=3)
        iou_masks = self.thresholding(iou_pair)
        # iou_masks:    (batch, #box, #box, #mask)

        score = score.transpose(dim0=1, dim1=2).unsqueeze(dim=3)
        adjc_scores = iou_masks * score
        max_idx = torch.argmax(adjc_scores, dim=1)
        # score:        (batch, #box, 1, 1)
        # adjc_scores:  (batch, #box, #box, #mask)
        # max_idx:      (batch, #box, #mask)

        box_indices = self.box_indices[:, :n_boxes]
        pool_masks = (max_idx == box_indices).long()
        pool_masks = pool_masks.transpose(dim0=1, dim1=2)
        # box_indices:  (1, #box, 1)
        # pool_masks:   (batch, #mask, #box)
        # print(box_indices.shape, pool_masks.shape)

        pool_weight = torch.sum(pool_masks * mask_weight, dim=1, keepdim=True)
        # mask_weight:  (batch, #mask, #box)
        # pool_weight:  (batch, 1, #box)
        return pool_weight


class WeightedMaxPool(nn.Module):
    def __init__(self, kernel_sizes=None):
        super(WeightedMaxPool, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 9, 15]
        self.kernel_sizes = kernel_sizes
        self.n_max_pooling = len(kernel_sizes)

        max_pools = list()
        for kernel_size in kernel_sizes:
            # max_pool(kernel_size, stride, padding, dilation)
            max_pool = nn.MaxPool2d(kernel_size, stride=1, padding=int(kernel_size / 2))
            max_pools.append(max_pool)
        self.max_pools = nn.ModuleList(*max_pools)

    def forward(self, mask_weight, ft_score):
        assert self.n_max_pooling == mask_weight.shape[1]

        ft_max_scores = torch.cat([max_pool(ft_score) for max_pool in self.max_pools], dim=1)
        ft_scores = torch.cat([ft_score] * self.n_max_pooling, dim=1)
        pool_masks = (ft_max_scores == ft_scores).long()
        pool_weight = torch.sum(pool_masks * mask_weight, dim=1, keepdim=True)
        return pool_weight


class LearnableMatrixNMS(nn.Module):
    def __init__(self, thresholds=None, max_boxes=2134):
        super(LearnableMatrixNMS, self).__init__()
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.4]
        self.sigmas = torch.from_numpy(np.array(thresholds)).view(1, 1, -1).cuda()
        self.n_masks = len(thresholds)
        self.max_boxes = max_boxes

    def forward(self, mask_weight, boxes, scores):
        # boxes:        (batch, 4, #boxes)
        # scores:       (batch, 1, #boxes)
        # feature:      (batch, #masks, #boxes)
        n_boxes = boxes.shape[2]

        gauss_decay = list()
        sort_indices = torch.argsort(scores[:, 0], dim=1, descending=True)
        for boxes_s, sort_indices_s in zip(boxes, sort_indices):
            sort_boxes_s = boxes_s.transpose(dim0=0, dim1=1)[sort_indices_s]
            # sort_indices_s:   (#boxes)
            # sort_boxes_s:     (#boxes, 4)

            iou_mat_s = lib_util.calc_jaccard_torch(sort_boxes_s, sort_boxes_s)
            iou_mat_s = iou_mat_s.triu(diagonal=1)
            # iou_mat_s:        (#boxes, #boxes)

            iou_max_s = torch.max(iou_mat_s, dim=0, keepdim=True)[0]
            iou_max_mat_s = torch.cat([iou_max_s] * n_boxes, dim=0)
            iou_max_mat_s = torch.transpose(iou_max_mat_s, dim0=0, dim1=1)
            # iou_max_s:        (1, #boxes)
            # iou_max_mat_s:    (#boxes, #boxes)

            gauss_decay_s = -1 * (iou_mat_s ** 2 - iou_max_mat_s ** 2)
            gauss_decay_s = torch.stack([gauss_decay_s] * self.n_masks, dim=2)
            # gauss_decay_s:    (#boxes, #boxes, #masks)

            gauss_decay_s = torch.exp(gauss_decay_s / (self.sigmas + 1e-12))
            gauss_decay_s = torch.min(gauss_decay_s, dim=0)[0]
            gauss_decay.append(gauss_decay_s)
            # gauss_decay_s:    (#boxes, #masks)

        gauss_decay = torch.stack(gauss_decay, dim=0).transpose(dim0=1, dim1=2)
        # gauss_decay:  (batch, #masks, #boxes)

        unsort_indices = sort_indices.argsort(dim=1)
        unsort_indices = torch.stack([unsort_indices] * self.n_masks, dim=1)
        gauss_decay = torch.gather(gauss_decay, dim=2, index=unsort_indices)
        # gauss_decay:  (batch, #masks, #boxes)

        # unsorted_boxes = sort_boxes_s.transpose(0, 1).unsqueeze(0)
        # unsorted_boxes = torch.gather(unsorted_boxes, dim=2, index=unsort_indices)

        lmax_scores = torch.sum(gauss_decay.detach() * mask_weight, dim=1, keepdim=True)
        # lmax_scores:  (batch, 1, #boxes)
        return lmax_scores


# class LearnableNMS(nn.Module):
#     def __init__(self, thresholds=None, max_boxes=2134):
#         super(LearnableNMS, self).__init__()
#         if thresholds is None:
#             thresholds = [0.1, 0.2, 0.4]
#         self.thresholds = thresholds
#         self.n_masks = len(thresholds)
#         self.max_boxes = max_boxes
#
#     def forward(self, mask_weight, boxes, scores):
#         # boxes:        (batch, 4, #boxes)
#         # scores:       (batch, 1, #boxes)
#         # feature:      (batch, #masks, #boxes)
#         batch_size = boxes.shape[0]
#         n_boxes = boxes.shape[2]
#
#
#         for boxes_s, scores_s in zip(boxes, scores):
#             for threshold in self.thresholds:
#
#
#
#
#         # masks:        (batch, #masks, #boxes)

