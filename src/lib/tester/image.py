import os
import cv2
from lib import util as lib_util
from . import __util__ as tester_util
from .__abc__ import TesterABC


class ImageTester(TesterABC):
    def __init__(self, global_args, tester_args):
        super(ImageTester, self).__init__(global_args, tester_args)
        self.coord_scale_factor = global_args['coord_scale_factor']
        self.img_size = (global_args['img_h'], global_args['img_w'])
        self.coord_range = (
            self.img_size[0] / self.coord_scale_factor,
            self.img_size[1] / self.coord_scale_factor,
        )
        self.img_size = (global_args['img_h'], global_args['img_w'])
        self.n_samples = tester_args['n_samples']
        self.max_boxes = tester_args['max_boxes']
        self.conf_thresh = tester_args['conf_thresh']

        self.mog_heatmap = tester_args['mog_heatmap']
        self.pi_thresh = tester_args['mog_heatmap_args']['pi_thresh']
        self.max_gauss = tester_args['mog_heatmap_args']['max_gauss']

    def run(self, framework, data_loader, result_dir):
        assert data_loader.batch_size == 1
        pre_proc = data_loader.dataset.pre_proc
        class_map = data_loader.dataset.get_number2name_map()
        lib_util.make_dir(result_dir)

        # -------------------
        # import torch
        # import numpy as np
        # img_dir = '/home/yoojy31/Desktop/test/shift_result3'
        # img_name_list = os.listdir(img_dir)
        # img_name_list.sort()
        # for i, img_name in enumerate(img_name_list):
        #     img_s = cv2.imread(os.path.join(img_dir, img_name))[:, :, ::-1]
        #     img = np.expand_dims(img_s.transpose(2, 0, 1), axis=0) / 255.0
        #     data_dict = dict()
        #     data_dict['img'] = torch.from_numpy(img)
        # -------------------

        for i, data_dict in enumerate(data_loader):
            output_dict, result_dict, _ = framework.infer_forward(data_dict)
            pred_boxes_s = lib_util.cvt_torch2numpy(result_dict['boxes_l'])[0]
            pred_confs_s = lib_util.cvt_torch2numpy(result_dict['confs_l'])[0]
            pred_labels_s = lib_util.cvt_torch2numpy(result_dict['labels_l'])[0]

            data_dict = pre_proc.inv_transform_batch(data_dict)
            img_s = data_dict['img'][0]
            gt_boxes_s = data_dict['boxes'][0]
            gt_labels_s = data_dict['labels'][0]

            sort_idx = 0
            gt_img_path = os.path.join(result_dir, '%03d_%d_%s.png' % (i, sort_idx, 'gt'))
            gt_img_s = tester_util.draw_boxes(
                img_s, gt_boxes_s, None, gt_labels_s,
                class_map, self.conf_thresh, self.max_boxes)
            cv2.imwrite(gt_img_path, gt_img_s[:, :, ::-1])
            # scipy.misc.imsave(gt_img_path, gt_img_s)
            sort_idx += 1

            # draw_boxes
            pred_img_path = os.path.join(result_dir, '%03d_%d_%s.png' % (i, sort_idx, 'pred'))
            pred_img_s = tester_util.draw_boxes(
                img_s, pred_boxes_s, pred_confs_s, pred_labels_s,
                class_map, self.conf_thresh, self.max_boxes)
            cv2.imwrite(pred_img_path, pred_img_s[:, :, ::-1])
            # scipy.misc.imsave(pred_img_path, pred_img_s)
            sort_idx += 1

            if self.mog_heatmap:
                mu, sig, pi = output_dict['mu'], output_dict['sig'], output_dict['pi']
                mog_heatmap = tester_util.draw_mog_heatmap(
                    mu, sig, pi, self.coord_range, self.img_size, self.pi_thresh, self.max_gauss)
                # mog_heatmap = scipy.misc.imresize(
                #     mog_heatmap.astype(np.uint8), self.img_size, interp='bilinear')

                mog_heatmap = tester_util.draw_boxes(
                    mog_heatmap, pred_boxes_s, pred_confs_s, pred_labels_s,
                    class_map, self.conf_thresh, self.max_boxes)
                mog_img_path = os.path.join(result_dir, '%03d_%d_%s.png' % (i, sort_idx, 'mog'))
                cv2.imwrite(mog_img_path, mog_heatmap[:, :, ::-1])
                # scipy.misc.imsave(mog_img_path, mog_heatmap)
                sort_idx += 1

                if 'mu_bar' in output_dict.keys():
                    mu_bar, sig_bar, pi_bar = \
                        output_dict['mu_bar'], output_dict['sig_bar'], output_dict['pi_bar']
                    mog_heatmap_bar = tester_util.draw_mog_heatmap(
                        mu_bar, sig_bar, pi_bar, self.coord_range, self.img_size,
                        self.pi_thresh, self.max_gauss)
                    # mog_heatmap_bar = scipy.misc.imresize(
                    #     mog_heatmap_bar.astype(np.uint8), self.img_size, interp='bilinear')

                    mog_heatmap_bar = tester_util.draw_boxes(
                        mog_heatmap_bar, pred_boxes_s, pred_confs_s, pred_labels_s,
                        class_map, self.conf_thresh, self.max_boxes)
                    mog_bar_img_path = os.path.join(result_dir, '%03d_%d_%s.png' % (i, sort_idx, 'mog_bar'))
                    cv2.imwrite(mog_bar_img_path, mog_heatmap_bar[:, :, ::-1])
                    # scipy.misc.imsave(mog_bar_img_path, mog_heatmap_bar)
                del mog_heatmap

            data_dict.clear()
            del data_dict
            if i >= (self.n_samples - 1):
                break
