from .voc import VOCDataset
from .coco import COCODataset


def get_dataset(dataset_key):
    return {
        'voc': VOCDataset,
        'coco': COCODataset,
    }[dataset_key]
