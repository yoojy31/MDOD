import numpy as np
from .__abc__ import PreProcABC
from lib import util as lib_util
from . import __util__ as pre_util


class BasePreProc(PreProcABC):
    def __init__(self, global_args, pre_proc_args):
        super(BasePreProc, self).__init__(global_args, pre_proc_args)
        self.rgb_mean = np.array(pre_proc_args['rgb_mean']).astype(np.float32).reshape(3, 1, 1)
        self.rgb_std = np.array(pre_proc_args['rgb_std']).astype(np.float32).reshape(3, 1, 1)
        self.pixel_scale = pre_proc_args['pixel_scale']
        self.augmentation = pre_proc_args['augmentation']
        self.xywh_box = global_args['xywh_box']

    def __augment__(self, sample_dict):
        img = np.array(sample_dict['img'])
        boxes = np.array(sample_dict['boxes'])
        labels = np.array(sample_dict['labels'])

        # img = pre_util.rand_brightness(img)
        # img = pre_util.rand_contrast(img)
        img, boxes = pre_util.expand(img, boxes)
        img, boxes, labels = pre_util.rand_crop(img, boxes, labels)
        img, boxes = pre_util.rand_flip(img, boxes)

        sample_dict['img'] = img
        sample_dict['boxes'] = boxes
        sample_dict['labels'] = labels
        return sample_dict

    def transform(self, sample_dict):
        s_dict = sample_dict
        s_dict['img'] = np.transpose(s_dict['img'], axes=(2, 0, 1)).astype(dtype=np.float32)
        s_dict['img'] *= (self.pixel_scale / 255.0)
        s_dict['img'] = (s_dict['img'] - self.rgb_mean) / self.rgb_std

        s_dict['boxes'] /= self.coord_scale_factor
        if self.xywh_box:
            s_dict['boxes'][:, [2, 3]] -= s_dict['boxes'][:, [0, 1]]
            s_dict['boxes'][:, [0, 1]] += (s_dict['boxes'][:, [2, 3]] * 0.5)
        s_dict['labels'] = np.expand_dims(s_dict['labels'], axis=1)
        return s_dict

    def inv_transform_batch(self, data_dict):
        d_dict = lib_util.cvt_torch2numpy(data_dict)
        d_dict['img'] = d_dict['img'] * self.rgb_std + self.rgb_mean
        d_dict['img'] = (np.transpose(d_dict['img'], axes=(0, 2, 3, 1))
                         * 255.0 / self.pixel_scale).astype(dtype=np.uint8)

        if self.xywh_box:
            d_dict['boxes'][:, :, [0, 1]] -= (d_dict['boxes'][:, :, [2, 3]] * 0.5)
            d_dict['boxes'][:, :, [2, 3]] += d_dict['boxes'][:, :, [0, 1]]
        d_dict['boxes'] *= self.coord_scale_factor
        d_dict['labels'] = np.squeeze(d_dict['labels'], axis=2)
        return d_dict

    def process(self, sample_dict):
        sample_dict['img'] = np.array(sample_dict['img']).astype(np.float32)
        sample_dict['boxes'] = np.array(sample_dict['boxes']).astype(np.float32)
        sample_dict['labels'] = np.array(sample_dict['labels']).astype(np.float32)

        s_dict = self.__augment__(sample_dict) if self.augmentation else sample_dict
        img_size = np.array(s_dict['img'].shape)[:2]
        s_dict['img'], s_dict['boxes'] = pre_util.resize(s_dict['img'], s_dict['boxes'], self.input_size)
        s_dict['boxes'] = lib_util.clip_boxes_s(s_dict['boxes'], self.input_size, numpy=True)

        n_boxes = s_dict['boxes'].shape[0]
        s_dict = self.transform(s_dict)
        s_dict.update({'n_boxes': np.array(n_boxes), 'img_size': np.array(img_size)})
        s_dict = self.__fill__(s_dict)
        return s_dict
