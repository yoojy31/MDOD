from .mdod import MDODNetwork

def get_network(network_key):
    return {
        'mdod': MDODNetwork,
    }[network_key]
