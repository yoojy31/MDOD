import copy
import numpy as np
import torch
import torch.nn.functional as func
from torchvision.ops.boxes import nms
from lib.network.loss_func import __util__ as loss_util
from lib import util as lib_util
import time


def calc_mog_nll(mu, sig, pi, boxes, n_boxes, coord_pdf=lib_util.gaussian_pdf, pi_thresh=None, value_return=False):
    mog_nll_loss = list()
    pred_gt_ratio = list()
    for i in range(mu.shape[0]):
        if n_boxes[i] == 0:
            pass
        else:
            mu_s, sig_s, pi_s = mu[i:i + 1], sig[i:i + 1], pi[i:i + 1]
            boxes_s = boxes[i:i + 1, :n_boxes[i]]

            if pi_thresh is not None:
                max_pi_s = torch.max(pi_s)
                norm_pi_s = pi_s / max_pi_s
                keep_idxes = torch.nonzero(norm_pi_s[0, 0] > pi_thresh).view(-1)
                mu_s, sig_s = mu_s[:, :, keep_idxes], sig_s[:, :, keep_idxes]
                pi_s = pi_s[:, :, keep_idxes]
                pi_s = pi_s / torch.sum(pi_s)

            pi_s = (n_boxes[i] * pi_s)
            mixture_lhs_s = lib_util.mm_pdf(
                mu_s, sig_s, pi_s, boxes_s, comp_pdf=coord_pdf, sum_comp=True)[0, :, 0]

            mixture_nll_s = -torch.log(mixture_lhs_s + lib_util.epsilon)
            mog_nll_loss.append(mixture_nll_s)
            del mixture_lhs_s

            if value_return:
                norm_pi_s = pi_s / torch.max(pi_s)
                keep_idxes = torch.nonzero(norm_pi_s[0, 0] > 0.001).view(-1)
                pred_gt_ratio.append((len(keep_idxes) / n_boxes[i].float()).view(1))

    mog_nll_loss = torch.cat(mog_nll_loss, dim=0)
    if value_return:
        pred_gt_ratio = torch.cat(pred_gt_ratio, dim=0)
        return mog_nll_loss, pred_gt_ratio
    else:
        return mog_nll_loss


def calc_mod_mm_nll(
    mu, sig, pi, prob, boxes, labels, n_boxes, n_samples, max_samples, pi_thresh,
    n_classes, coord_pdf=lib_util.gaussian_pdf, sampling_noise=True):

    cmps_pi = pi * n_boxes.view(n_boxes.shape[0], 1, 1)
    bg_labels_s = torch.zeros((torch.max(n_boxes) * n_samples, n_classes)).float().cuda()
    bg_labels_s[:, 0] = 1.0
    labels = func.one_hot(labels[:, :, 0], n_classes).float()
    sample_boxes = lib_util.sample_coords_from_mog(
        mu, sig, pi, min(int(torch.max(n_boxes) * n_samples), max_samples), sampling_noise=sampling_noise)

    mod_mm_nll_loss = list()
    for mu_s, sig_s, prob_s, pi_s, boxes_s, labels_s, sample_boxes_s, n_boxes_s in \
            zip(mu, sig, prob, cmps_pi, boxes, labels, sample_boxes, n_boxes):
        boxes_s, labels_s = boxes_s[:n_boxes_s], labels_s[:n_boxes_s]

        if pi_thresh is not None:
            max_pi_s = torch.max(pi_s)
            norm_pi_s = pi_s / max_pi_s
            keep_idxes = torch.nonzero(norm_pi_s[0] > pi_thresh).view(-1)
            mu_s, sig_s = mu_s[:, keep_idxes], sig_s[:, keep_idxes]
            pi_s, prob_s = pi_s[:, keep_idxes], prob_s[:, keep_idxes]

        if n_boxes_s <= 0:
            sample_boxes_s = sample_boxes_s[:1]
            sample_labels_s = bg_labels_s[:1]
        else:
            sample_boxes_s = sample_boxes_s[:n_boxes_s * n_samples]
            iou_pairs = lib_util.calc_jaccard_torch(sample_boxes_s, boxes_s)
            max_ious, argmax_ious = torch.max(iou_pairs, dim=1)
            sample_labels_s = labels_s[argmax_ious]
            sample_labels_s = torch.where(
                max_ious.unsqueeze(dim=1) > 0.5, sample_labels_s,
                bg_labels_s[:sample_boxes_s.shape[0]])

        gauss_lhs_s = lib_util.mm_pdf_s(
            mu_s, sig_s, pi_s, sample_boxes_s, comp_pdf=coord_pdf, sum_comp=False)
        cat_probs_s = lib_util.category_pmf_s(prob_s, sample_labels_s.float())
        mm_lhs_s = torch.sum(gauss_lhs_s * cat_probs_s, dim=1)

        mm_nll_s = -torch.log(mm_lhs_s + lib_util.epsilon)
        mod_mm_nll_loss.append(mm_nll_s)

    mod_mm_nll_loss = torch.cat(mod_mm_nll_loss, dim=0)
    return mod_mm_nll_loss


def calc_max_nll(mu, sig, pi, boxes, n_boxes):
    mog_nll_loss = list()
    for i in range(mu.shape[0]):
        if n_boxes[i] <= 0:
            pass
        else:
            mu_s, sig_s, pi_s = mu[i:i + 1], sig[i:i + 1], pi[i:i + 1]
            boxes_s = boxes[i:i + 1, :n_boxes[i]]

            pi_s = n_boxes[i] * pi_s
            mixture_lhs_s = lib_util.mm_pdf(mu_s, sig_s, pi_s, boxes_s, sum_comp=False)[0, :, 0]
            mixture_lhs_s = torch.max(mixture_lhs_s, dim=1)[0]
            mixture_nll_s = -torch.log(mixture_lhs_s + lib_util.epsilon)
            mog_nll_loss.append(mixture_nll_s)

    mog_nll_loss = torch.cat(mog_nll_loss, dim=0)
    return mog_nll_loss


def calc_cluster_nll(mu, sig, pi, boxes, n_boxes, coord_pdf=lib_util.gaussian_pdf):
    cluster_nll_loss = list()

    for i in range(mu.shape[0]):
        if n_boxes[i] == 0:
            pass
        else:
            pi_s = n_boxes[i] * pi[i:i + 1]
            mu_s, sig_s = mu[i:i + 1], sig[i:i + 1]
            boxes_s = boxes[i:i + 1, :n_boxes[i]]

            comp_lhs_s = lib_util.mm_pdf(
                mu_s, sig_s, pi_s, boxes_s, comp_pdf=coord_pdf, sum_comp=False)[0, :, 0]
            mixture_lhs_s = torch.sum(comp_lhs_s, dim=1)
            max_lhs_s = torch.max(comp_lhs_s, dim=1)[0]

            cluster_nll_s = -torch.log(max_lhs_s / (mixture_lhs_s + lib_util.epsilon) + lib_util.epsilon)
            cluster_nll_loss.append(cluster_nll_s)
            del cluster_nll_s

    cluster_nll_loss = torch.cat(cluster_nll_loss, dim=0)
    return cluster_nll_loss
