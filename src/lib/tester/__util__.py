import numpy as np
import cv2
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lib import util as lib_util

colors = ((64, 64, 64), (31, 119, 180), (174, 199, 232), (255, 127, 14),
          (255, 187, 120), (44, 160, 44), (152, 223, 138), (214, 39, 40),
          (255, 152, 150), (148, 103, 189), (197, 176, 213), (140, 86, 75),
          (196, 156, 148), (227, 119, 194), (247, 182, 210), (127, 127, 127),
          (199, 199, 199), (188, 189, 34), (219, 219, 141), (23, 190, 207),
          (158, 218, 229), (180, 119, 31))


def draw_rec_prec_graph(recall, precision):
    plt.clf()
    intv = int(len(recall) / 500)
    rec_prec = pd.DataFrame(
        {'recall': recall[::intv], 'precision': precision[::intv]})
    sns_plot = sns.lineplot(x='recall', y='precision', data=rec_prec)
    sns_plot.set(xlim=(0.0, 1.05), ylim=(0.0, 1.05))
    plt.grid(True)
    figure = sns_plot.figure
    return figure


def draw_boxes(img_s, boxes_s, confs_s=None, labels_s=None,
               class_map=None, conf_thresh=0.0, max_boxes=100):

    box_img_s = img_s.copy()
    n_draw_boxes = 0
    n_wrong_boxes = 0
    n_thresh_boxes = 0
    for i, box in enumerate(boxes_s):
        try:
            l, t = int(round(box[0])), int(round(box[1]))
            r, b = int(round(box[2])), int(round(box[3]))
        except IndexError:
            print(boxes_s)
            print(i, box)
            print('IndexError')
            exit()

        if confs_s is not None:
            # if (0.05 > confs_s[i]) or (confs_s[i] > 0.5):
            # if confs_s[i] > 0.1:
            if conf_thresh > confs_s[i]:
                n_thresh_boxes += 1
                continue
        if (r - l <= 0) or (b - t <= 0):
            n_wrong_boxes += 1
            continue
        if n_draw_boxes >= max_boxes:
            continue

        conf_str = '-' if confs_s is None else '%0.3f' % confs_s[i]
        if labels_s is None:
            lab_str, color = '-', colors[i % len(colors)]
        else:
            lab_i = int(labels_s[i])
            lab_str = str(lab_i) if class_map is None else class_map[lab_i]
            color = colors[lab_i % len(colors)]

        box_img_s = cv2.rectangle(box_img_s, (l, t), (r, b), color, 2)
        l = int(l - 1 if l > 1 else r - 60)
        t = int(t - 8 if t > 8 else b)
        r, b = int(l + 60), int(t + 8)
        box_img_s = cv2.rectangle(box_img_s, (l, t), (r, b), color, cv2.FILLED)
        box_img_s = cv2.putText(box_img_s, '%s %s' % (conf_str, lab_str), (l + 1, t + 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255),
                                1, cv2.LINE_AA)
        n_draw_boxes += 1

    info_text = 'n_draw_b: %d, n_thr_b: %d, n_wrong_b: %d' % \
                (n_draw_boxes, n_thresh_boxes, n_wrong_boxes)
    if confs_s is not None:
        info_text += ', sum_of_conf: %.3f' % (np.sum(confs_s))
    else:
        info_text += ', sum_of_conf: -'

    box_img_s = cv2.rectangle(box_img_s, (0, 0), (350, 11), (0, 0, 0), cv2.FILLED)
    box_img_s = cv2.putText(box_img_s, info_text, (5, 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.25, (255, 255, 255), 1, cv2.LINE_AA)
    return box_img_s


def draw_points(img_s, points_s, confs_s=None, max_points=100):
    point_img_s = img_s.copy()
    if len(point_img_s.shape) == 2:
        point_img_s = np.stack([point_img_s] * 3, axis=2)

    for i, point in enumerate(points_s):
        if i >= max_points:
            break
        center, radian, color = (point[0], point[1]), 2, colors[i % len(colors)]
        # if confs[i] < 0.05:
        #     continue
        point_img_s = cv2.circle(point_img_s, center, radian, color, 1)
        if confs_s is not None:
            point_img_s = cv2.putText(point_img_s, '%.3f' % confs_s[i], center,
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.25, (255, 255, 255), 1, cv2.LINE_AA)
    return point_img_s


def draw_mog_heatmap(mu, sig, pi, coord_size, size, pi_thresh=0.001, max_gauss=100):
    torch.autograd.set_grad_enabled(False)
    mu1, mu2, sig1, sig2 = mu[:, :2], mu[:, 2:], sig[:, :2], sig[:, 2:]

    points = lib_util.create_coord_map(size, coord_size)
    points = points.reshape((1, 2, -1)).transpose((0, 2, 1))
    # points.shape: (batch, #points, 2)

    points = torch.from_numpy(points).float().cuda(mu1.device)
    # print(mu1.shape, sig1.shape, pi.shape, points.shape, torch.min(points), torch.max(points))
    # print(torch.min(mu1), torch.max(mu1), torch.min(mu2), torch.max(mu2))
    # print('')
    lhs1 = lib_util.cvt_torch2numpy(lib_util.mm_pdf(mu1, sig1, pi, points.clone()))
    lhs2 = lib_util.cvt_torch2numpy(lib_util.mm_pdf(mu2, sig2, pi, points.clone()))

    h, w = size
    lh_map1, lh_map2 = lhs1.reshape((h, w, 1)), lhs2.reshape((h, w, 1))
    lh_map1 = lh_map1 * 255 / np.max(lh_map1)
    lh_map2 = lh_map2 * 255 / np.max(lh_map2)

    # QUANTIZE
    quant_factor = 7
    lh_map1 = (lh_map1 * quant_factor / 255).astype(np.int).astype(float) * 255 / quant_factor
    lh_map2 = (lh_map2 * quant_factor / 255).astype(np.int).astype(float) * 255 / quant_factor

    mog_heatmap = np.concatenate(
        [lh_map1 + lh_map2 * (255 / 255), lh_map2 * (127 / 255), np.ones((h, w, 1)) * 32], axis=2
    ).astype(np.uint8)

    boxes_s, mog_confs_s = mu.transpose(1, 2)[0], pi.transpose(1, 2)[0]
    boxes_s, mog_confs_s = lib_util.sort_boxes_s(boxes_s, mog_confs_s)
    boxes_s, mog_confs_s = boxes_s[:max_gauss], mog_confs_s[:max_gauss]

    pi_s = mog_confs_s / torch.max(mog_confs_s)
    keep_idxes = torch.nonzero(pi_s > pi_thresh).view(-1)
    boxes_s = boxes_s[keep_idxes]
    mog_confs_s = mog_confs_s[keep_idxes]

    boxes_s[:, [0, 2]] *= (size[1] / coord_size[1])
    boxes_s[:, [1, 3]] *= (size[0] / coord_size[0])

    mog_heatmap = draw_points(mog_heatmap, boxes_s[:, :2], mog_confs_s, max_gauss)
    mog_heatmap = draw_points(mog_heatmap, boxes_s[:, 2:], mog_confs_s, max_gauss)
    del points
    return mog_heatmap
