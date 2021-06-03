#!/usr/bin/env bash

python3 ./src/run.py \
--bash_file="./run/run_mdod_coco.sh" \
--result_dir="./result/mdod/coco/`(date "+%Y%m%d-%H%M%S")`_320x320_res50" \
\
--run_args="{
    'devices': [0], 'port': 12355, 'sync_bnorm': False,
    'max_grad': 7, 'grad_accum': 1, 'amp': False,

    'init_epoch': 0, 'max_epoch': 160, 'print_intv': 10,
    'lr_decay_schd': {120: 0.1, 150: 0.1},
    'loss_args_schd': {},
    'test_epoch_list': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
    'save_epoch_list': [20, 40, 60, 80, 100, 120, 140, 150, 160],

    'write_log': False, 'load_dir': None,
    'manually_shuffle': True,
}" \
--global_args="{
    'n_classes': 81, 'img_h': 320, 'img_w': 320, 'coord_scale_factor': 32,  'xywh_box': False,
}" \
--framework_info="{
    'framework': 'basic',
    'framework_args': {},
}" \
--network_info="{
    'network': 'mdod', 'network_args': {
        'version_old': False, 'actv_func': 'swish',
        'output_sizes': None, 'std_factor': 0.05,
        'max_batch_size': 32, 'num_filters': 256,

        'ft_extractor': 'res50fpn',
        'ft_extractor_args': {
            'pretrained': True, 'num_filters': 256,
            'upsampling': 'nearest', 'fpn_actv_func': 'relu',
            'group_norm': False, 'group_norm_size': 32,
            'init_method': None,
        },

        'loss_func': 'mdod',
        'loss_func_args': {
            'lw_dict': {'mog_nll': 1.0, 'mod_nll': 2.0},
            'mod_n_samples': 5, 'mog_pi_thresh': None, 'mod_pi_thresh': None,
            'mod_max_samples': 150, 'coord_pdf': 'cauchy', 'sampling_noise': False, 
            'value_return': False,
        },
    },
}" \
--post_proc_info="{
    'post_proc': 'mdod_tf',
    'post_proc_args': {
        'pi_thresh': 0.001, 'conf_thresh': 0.001,
        'nms_type': 'tf_greedy', 'nms_thresh': 0.5,
        'max_boxes': 500,
    },
}" \
--optimizer_info="{
    'optimizer': 'sgd',
    'optimizer_args': {
        'lr': 0.005, 'momentum': 0.9, 'weight_decay': 0.00005,
    },
}" \
--train_data_loader_info="{
    'dataset': 'coco',
    'dataset_args': {
        'roots': ['./data/coco2017'],
        'types': ['train'],
        'pre_proc': 'base',
        'pre_proc_args': {
            'augmentation': True, 'max_boxes': 300,
            'rgb_mean': [0.485, 0.456, 0.406],
            'rgb_std': [0.229, 0.224, 0.225],
            'pixel_scale': 1,
        },
        'grouping': False,
    },
    'shuffle': False, 'num_workers': 4, 'batch_size': 32,
}" \
--test_data_loader_info="{
    'dataset': 'coco',
    'dataset_args': {
        'roots': ['./data/coco2017'],
        'types': ['val'],
        'pre_proc': 'base',
        'pre_proc_args': {
            'augmentation': False, 'max_boxes': 300,
            'rgb_mean': [0.485, 0.456, 0.406],
            'rgb_std': [0.229, 0.224, 0.225],
            'pixel_scale': 1,
        },
        'grouping': False,
    },
    'shuffle': False, 'num_workers': 2, 'batch_size': 1,
}" \
--tester_info_list="[{
    'tester': 'image',
    'tester_args': {
        'n_samples': 50, 'conf_thresh': 0.2, 'max_boxes': 50,
        'mog_heatmap': False,
        'mog_heatmap_args': {'pi_thresh': 0.001, 'max_gauss': 100},
    },
}, {
    'tester': 'quant',
    'tester_args': {'dataset': 'coco_2017_val'}
}]"

