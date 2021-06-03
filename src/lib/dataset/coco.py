import math
import cv2
import random
import numpy as np
from copy import deepcopy
from lib.external.dataset.roidb import combined_roidb
from .__abc__ import DatasetABC


class COCODataset(DatasetABC):
    def __init__(self, global_args, dataset_args):
        super(COCODataset, self).__init__(global_args, dataset_args)
        imdb_names = "coco_2017_" + dataset_args['types'][0]
        self.roidb = combined_roidb(imdb_names, self.roots[0])

        self.copy_roidb = deepcopy(self.roidb)
        self.data_size = len(self.roidb)

        self.number2name_map = {
            0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
            5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
            11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench', 15: 'bird',
            16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow', 21: 'elephant',
            22: 'bear', 23: 'zebra', 24: 'giraffe', 25: 'backpack', 26: 'umbrella',
            27: 'handbag', 28: 'tie', 29: 'suitcase', 30: 'frisbee', 31: 'skis', 32: 'snowboard',
            33: 'sports ball', 34: 'kite', 35: 'baseball bat', 36: 'baseball glove',
            37: 'skateboard', 38: 'surfboard', 39: 'tennis racket', 40: 'bottle',
            41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife', 45: 'spoon', 46: 'bowl',
            47: 'banana', 48: 'apple', 49: 'sandwich', 50: 'orange', 51: 'broccoli',
            52: 'carrot', 53: 'hot dog', 54: 'pizza', 55: 'donut', 56: 'cake',
            57: 'chair', 58: 'couch', 59: 'potted plant', 60: 'bed', 61: 'dining table',
            62: 'toilet', 63: 'tv', 64: 'laptop', 65: 'mouse', 66: 'remote', 67: 'keyboard',
            68: 'cell phone', 69: 'microwave', 70: 'oven', 71: 'toaster', 72: 'sink',
            73: 'refrigerator', 74: 'book', 75: 'clock', 76: 'vase', 77: 'scissors',
            78: 'teddy bear', 79: 'hair drier', 80: 'toothbrush'}

        self.name2number_map = {
            'background': 0, 'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4,
            'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10,
            'fire hydrant': 11, 'stop sign': 12, 'parking meter': 13, 'bench': 14, 'bird': 15,
            'cat': 16, 'dog': 17, 'horse': 18, 'sheep': 19, 'cow': 20, 'elephant': 21, 'bear': 22,
            'zebra': 23, 'giraffe': 24, 'backpack': 25, 'umbrella': 26, 'handbag': 27,
            'tie': 28, 'suitcase': 29, 'frisbee': 30, 'skis': 31, 'snowboard': 32,
            'sports ball': 33, 'kite': 34, 'baseball bat': 35, 'baseball glove': 36,
            'skateboard': 37, 'surfboard': 38, 'tennis racket': 39, 'bottle': 40,
            'wine glass': 41, 'cup': 42, 'fork': 43, 'knife': 44, 'spoon': 45, 'bowl': 46,
            'banana': 47, 'apple': 48, 'sandwich': 49, 'orange': 50, 'broccoli': 51,
            'carrot': 52, 'hot dog': 53, 'pizza': 54, 'donut': 55, 'cake': 56,
            'chair': 57, 'couch': 58, 'potted plant': 59, 'bed': 60, 'dining table': 61,
            'toilet': 62, 'tv': 63, 'laptop': 64, 'mouse': 65, 'remote': 66, 'keyboard': 67,
            'cell phone': 68, 'microwave': 69, 'oven': 70, 'toaster': 71, 'sink': 72,
            'refrigerator': 73, 'book': 74, 'clock': 75, 'vase': 76, 'scissors': 77,
            'teddy bear': 78, 'hair drier': 79, 'toothbrush': 80}

    def __getitem__(self, index):
        minibatch_db = self.roidb[index]
        img = cv2.imread(minibatch_db['image'])[:, :, ::-1]

        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
        boxes = minibatch_db['boxes']
        labels = minibatch_db['gt_classes']

        sample_dict = {'img': img, 'boxes': boxes, 'labels': labels}
        sample_dict = self.pre_proc.process(sample_dict)
        return sample_dict

    def __len__(self):
        return len(self.roidb)

    def shuffle(self, seed=0):
        random.seed(seed)
        self.roidb = deepcopy(self.copy_roidb)
        random.shuffle(self.roidb)

    def get_number2name_map(self):
        return self.number2name_map

    def get_name2number_map(self):
        return self.name2number_map

    def get_dataset_roots(self):
        return self.roots
