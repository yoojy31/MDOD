import numpy as np
import torch
import tensorflow as tf
from lib import util as lib_util
from .__abc__ import PostProcABC


class MDODPostProcTF(PostProcABC):
    def __init__(self, global_args, post_proc_args):
        super(MDODPostProcTF, self).__init__(global_args, post_proc_args)
        self.input_size = (global_args['img_h'], global_args['img_w'])
        self.coord_scale_factor = global_args['coord_scale_factor']
        self.n_classes = global_args['n_classes']

        self.pi_thresh = post_proc_args['pi_thresh']
        self.conf_thresh = post_proc_args['conf_thresh']
        self.nms_thresh = post_proc_args['nms_thresh']
        self.max_boxes = post_proc_args['max_boxes']

    def process(self, output_dict, data_dict):
        pi_s, mu_s, prob_s = output_dict['pi'], output_dict['mu'], output_dict['prob']
        input_size = data_dict['img'].shape[2:4]
        assert mu_s.shape[0] == 1

        boxes_s = mu_s.transpose(1, 2)
        boxes_s *= self.coord_scale_factor
        boxes_s = lib_util.clip_boxes(boxes_s, input_size)

        boxes_s = boxes_s.unsqueeze(dim=1)
        confs_s = prob_s[:, 1:] if self.n_classes == prob_s.shape[1] else prob_s

        if self.pi_thresh is not None:
            norm_pi_s = pi_s / torch.max(pi_s)
            keep_idxes = torch.nonzero(norm_pi_s[0, 0] > self.pi_thresh).view(-1)
            boxes_s = boxes_s[:, :, keep_idxes]
            confs_s = confs_s[:, :, keep_idxes]

        boxes_s = boxes_s.transpose(1, 2)
        confs_s = confs_s.transpose(1, 2)
        boxes_s = boxes_s.detach().cpu().numpy()
        confs_s = confs_s.detach().cpu().numpy()

        boxes_s = tf.convert_to_tensor(boxes_s)
        confs_s = tf.convert_to_tensor(confs_s)
        boxes_s, confs_s, labels_s, n_dets = tf.image.combined_non_max_suppression(
            boxes_s, confs_s, self.max_boxes, self.max_boxes, self.nms_thresh,
            self.conf_thresh, clip_boxes=False)

        boxes_l = [boxes_s[0, :n_dets[0]].numpy()]
        confs_l = [confs_s[0, :n_dets[0]].numpy()]
        labels_l = [labels_s[0, :n_dets[0]].numpy() + 1.0]
        return {'boxes_l': boxes_l, 'confs_l': confs_l, 'labels_l': labels_l}, {}
