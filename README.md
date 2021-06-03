Mixture Density Object Detector: PyTorch Implementation
===========================================================

This repository is the PyTorch implementation of  
"[Density-based Object Detection: Learning Bounding Boxes without Ground Truth Assignment](https://arxiv.org/abs/1911.12721)"

Environment
-----------
- python3
- pytorch1.7

Directory Structure
-------------------
```
(root-directory)
├── README.md
├── run_mmod.py
├── src
│   └── (python-source-file.py)
├── result
│   └── (result-directory)
│       ├── ...
│       └── snapshot
│           └── (iteration)
│               ├── network.pth
│               └── optimizer.pth
└── data
    └── coco-2017
        ├── annotations
        └── images
```
You can download the voc and coco dataset in the follow links.  
http://cocodataset.org/#download (coco-2017)

Usage
-----
Training
```
# run_mmod.sh
--training_args="{'max_iter': maximum number of iterations, ...}"

# command
.../(root-directory)$ bash run_mmod.sh
```
Test
```
# run_mmod.sh
--training_args="{'init_iter': 0, 'max_iter': 0, ...}",
--test_iters="[0]"
--load_dir="path of the snapshot directory that has a network.pth file"

# command
.../(root-directory)$ bash run_mmod.sh
```

Citation
--------
```
@article{yoo2019density,
  title={Density-based Object Detection: Learning Bounding Boxes without Ground Truth Assignment},
  author={Yoo, Jaeyoung and Lee, Hojun and Chung, Inseop and Seo, Geonseok and Kwak, Nojun},
  journal={arXiv preprint arXiv:1911.12721},
  year={2019}
}
