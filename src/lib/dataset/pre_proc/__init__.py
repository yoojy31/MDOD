from .base import BasePreProc


def get_pre_proc(pre_proc_key):
    return {
        'base': BasePreProc,
        None: None
    }[pre_proc_key]
