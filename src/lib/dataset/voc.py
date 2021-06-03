import os
import random
import cv2
import PIL.Image as pilimg
import numpy as np
import scipy.misc
from xml.etree import ElementTree
from .__abc__ import DatasetABC


class VOCDataset(DatasetABC):
    def __init__(self, global_args, dataset_args):
        super(VOCDataset, self).__init__(global_args, dataset_args)
        img_pathes = list()
        anno_pathes = list()
        for root_dir, set_type in zip(self.roots, self.types):
            set_path = os.path.join(root_dir, 'ImageSets', 'Main', '%s.txt' % set_type)
            img_path_form = os.path.join(root_dir, 'JPEGImages', '%s.jpg')
            anno_path_form = os.path.join(root_dir, 'Annotations', '%s.xml')

            with open(set_path) as file:
                for img_name in file.readlines():
                    img_name = img_name.strip('\n')
                    img_pathes.append(img_path_form % img_name)
                    anno_pathes.append(anno_path_form % img_name)

        self.img_pathes = np.array(img_pathes).astype(np.string_)
        self.anno_pathes = np.array(anno_pathes).astype(np.string_)

        self.name2number_map = {
            'background': 0,
            'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4,
            'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8, 'chair': 9,
            'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13,
            'motorbike': 14, 'person': 15, 'pottedplant': 16,
            'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}
        self.number2name_map = {
            0: 'background',
            1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
            5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
            10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
            14: 'motorbike', 15: 'person', 16: 'pottedplant',
            17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

    def __len__(self):
        return len(self.img_pathes)

    # def shuffle(self):
    #     random.shuffle(self.roidb)

    def __getitem__(self, data_idx):
        # img = scipy.misc.imread(self.img_pathes[data_idx])
        # img = cv2.imread(self.img_pathes[data_idx])[:, :, ::-1]
        # print(self.img_pathes[data_idx])
        # img = cv2.imread(self.img_pathes[data_idx], 1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = pilimg.open(self.img_pathes[data_idx])
        anno = ElementTree.parse(self.anno_pathes[data_idx]).getroot()
        boxes, labels = self.__parse_anno__(anno)

        sample_dict = {'img': img, 'boxes': boxes, 'labels': labels}
        sample_dict = self.pre_proc.process(sample_dict)
        sample_dict['name'] = str(self.img_pathes[data_idx]).split('/')[-1][:-1]
        return sample_dict

    def __parse_anno__(self, anno):
        boxes = list()
        labels = list()
        for obj in anno.findall('object'):
            bndbox = obj.find('bndbox')
            boxes.append([
                float(bndbox.find('xmin').text), float(bndbox.find('ymin').text),
                float(bndbox.find('xmax').text), float(bndbox.find('ymax').text)])
            labels.append(np.array(self.name2number_map[obj.find('name').text]))
        boxes = np.array(boxes)
        labels = np.array(labels)
        return boxes, labels

    def get_name2number_map(self):
        return self.name2number_map

    def get_number2name_map(self):
        return self.number2name_map

    def get_dataset_roots(self):
        return self.roots
