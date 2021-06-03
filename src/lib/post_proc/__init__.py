from .mdod import MDODPostProcTF


def get_post_proc(key):
    return {
        'mdod_tf': MDODPostProcTF,
        None: None
    }[key]
