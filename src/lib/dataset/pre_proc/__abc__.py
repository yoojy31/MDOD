import abc
import numpy as np


class PreProcABC(abc.ABC):
    def __init__(self, global_args, pre_proc_args):
        self.n_classes = global_args['n_classes']
        self.input_size = (global_args['img_h'], global_args['img_w'])
        self.coord_scale_factor = global_args['coord_scale_factor']
        self.max_boxes = pre_proc_args['max_boxes']

    def __fill__(self, sample_dict):
        def create_dummy_boxes(_n_dummies):
            boxes = list()
            labels = list()
            for _ in range(_n_dummies):
                boxes.append(np.array([0, 0, 0, 0]))
                labels.append(np.array([0]))
            return np.array(boxes), np.array(labels)

        n_boxes = sample_dict['boxes'].shape[0]
        n_dummies = self.max_boxes - n_boxes

        if n_dummies > 0:
            dummy_boxes, dummy_labels = create_dummy_boxes(n_dummies)
            sample_dict['boxes'] = np.concatenate((sample_dict['boxes'], dummy_boxes), axis=0)
            sample_dict['boxes'] = sample_dict['boxes'].astype(np.float32)
            if 'labels' in sample_dict.keys():
                sample_dict['labels'] = np.concatenate((sample_dict['labels'], dummy_labels), axis=0)
                sample_dict['labels'] = sample_dict['labels'].astype(np.float32)
        else:
            sample_dict['boxes'] = sample_dict['boxes'][:self.max_boxes]
            sample_dict['boxes'] = sample_dict['boxes'].astype(np.float32)
            if 'labels' in sample_dict.keys():
                sample_dict['labels'] = sample_dict['labels'][:self.max_boxes]
                sample_dict['labels'] = sample_dict['labels'].astype(np.float32)
        return sample_dict

    @ abc.abstractmethod
    def __augment__(self, sample_dict):
        pass

    @ abc.abstractmethod
    def transform(self, sample_dict):
        pass

    @ abc.abstractmethod
    def inv_transform_batch(self, data_dict):
        pass

    @ abc.abstractmethod
    def process(self, sample_dict):
        pass
