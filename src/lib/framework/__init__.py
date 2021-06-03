from .basic import BasicFramework


def get_framework(framework_key):
    return {
        'basic': BasicFramework,
    }[framework_key]
