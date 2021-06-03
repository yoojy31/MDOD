import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from lib.post_proc import get_post_proc
from lib.framework import get_framework
from lib.network import get_network
from lib.dataset import get_dataset
from lib.tester import get_tester


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bash_file', type=str)
    parser.add_argument('--result_dir', type=str)

    parser.add_argument('--run_args', type=str)
    parser.add_argument('--global_args', type=str)
    parser.add_argument('--framework_info', type=str)
    parser.add_argument('--network_info', type=str)
    parser.add_argument('--post_proc_info', type=str)
    parser.add_argument('--optimizer_info', type=str)

    parser.add_argument('--train_data_loader_info', type=str)
    parser.add_argument('--test_data_loader_info', type=str)
    parser.add_argument('--tester_info_list', type=str)

    args = parser.parse_args()
    args.global_args = cvt_str2python_data(args.global_args)

    args.run_args = cvt_str2python_data(args.run_args)
    args.framework_info = cvt_str2python_data(args.framework_info)
    args.network_info = cvt_str2python_data(args.network_info)
    args.post_proc_info = cvt_str2python_data(args.post_proc_info)
    args.optimizer_info = cvt_str2python_data(args.optimizer_info)

    args.train_data_loader_info = cvt_str2python_data(args.train_data_loader_info)
    args.test_data_loader_info = cvt_str2python_data(args.test_data_loader_info)
    args.tester_info_list = cvt_str2python_data(args.tester_info_list)

    args.result_dict_dict = dict()
    args.result_dir_dict = dict()
    args.result_dir_dict['root'] = args.result_dir
    args.result_dir_dict['src'] = os.path.join(args.result_dir, 'src')
    args.result_dir_dict['log'] = os.path.join(args.result_dir, 'log')
    args.result_dir_dict['test'] = os.path.join(args.result_dir, 'test')
    args.result_dir_dict['snapshot'] = os.path.join(args.result_dir, 'snapshot')
    return args


def cvt_str2python_data(arg_str):
    if isinstance(arg_str, str):
        python_data = yaml.full_load(arg_str)
    else:
        python_data = arg_str

    if isinstance(python_data, dict):
        for key, value in python_data.items():
            if value == 'None':
                python_data[key] = None
            elif isinstance(value, dict) or isinstance(value, list):
                python_data[key] = cvt_str2python_data(value)

    elif isinstance(python_data, list):
        for i, value in enumerate(python_data):
            if value == 'None':
                python_data[i] = None
            elif isinstance(value, dict) or isinstance(value, list):
                python_data[i] = cvt_str2python_data(value)
    return python_data


def create_network(global_args, network_info):
    network_key = network_info['network']
    network_args = network_info['network_args']
    network = get_network(network_key)(global_args, network_args)
    network.build()
    network.cuda()
    return network


def create_post_proc(global_args, post_proc_info):
    post_proc_key = post_proc_info['post_proc']
    post_proc_args = post_proc_info['post_proc_args']
    return get_post_proc(post_proc_key)(global_args, post_proc_args)


def create_framework(global_args, framework_info, network, post_proc, world_size):
    framework_key = framework_info['framework']
    framework_args = framework_info['framework_args']
    return get_framework(framework_key)(global_args, framework_args, network, post_proc, world_size)


def create_optimizer(optimizer_info, network):
    optimizer_dict = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
    }

    optimizer_key = optimizer_info['optimizer']
    optimizer_args = optimizer_info['optimizer_args']
    optimizer_args.update({'params': network.parameters()})
    return optimizer_dict[optimizer_key](**optimizer_args)


def create_dataset(global_args, data_loader_info):
    dataset_key = data_loader_info['dataset']
    dataset_args = data_loader_info['dataset_args']
    dataset = get_dataset(dataset_key)(global_args, dataset_args)
    return dataset


def create_data_loader(dataset, data_loader_info, rank=0, world_size=1):
    batch_size = data_loader_info['batch_size']
    num_workers = data_loader_info['num_workers']
    shuffle = data_loader_info['shuffle']
    img_per_gpu = int(batch_size / world_size)

    sampler = None
    is_distributed = True if world_size > 1 else False
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=shuffle, num_replicas=world_size, rank=rank)
        shuffle = False

    data_loader = DataLoader(
        dataset=dataset, batch_size=img_per_gpu, shuffle=shuffle,
        num_workers=num_workers, sampler=sampler)
    return data_loader


def create_tester(global_args, tester_info):
    tester_key = tester_info['tester']
    tester_args = tester_info['tester_args']
    return get_tester(tester_key)(global_args, tester_args)
