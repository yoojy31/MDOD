from .mdod import MDODLossFunc


def get_loss_func(loss_func_key):
    return {
        'mdod': MDODLossFunc,
        None: None
    }[loss_func_key]
