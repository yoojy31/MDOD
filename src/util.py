import os
import shutil
# import torch
# import torch.nn as nn
from lib import util as lib_util


def cvt_dict2str(value_dict):
    result_str = ''
    for key, value in value_dict.items():
        result_str += ('%s: %.5f, ' % (key, value))
    result_str = result_str[:-2]
    return result_str.rstrip()


def create_result_dir(result_dir, names=None):
    result_dir_dict = dict()
    lib_util.make_dir(result_dir)
    for name in names:
        dir_path = os.path.join(result_dir, name)
        lib_util.make_dir(dir_path)
        result_dir_dict[name] = dir_path
    return result_dir_dict


def copy_file(src_path, dst_dir):
    src_file = src_path.split('/')[-1]
    dst_path = os.path.join(dst_dir, src_file)
    shutil.copyfile(src_path, dst_path)


def copy_dir(src, dst, symlinks=False, ignore=None):
    lib_util.make_dir(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
