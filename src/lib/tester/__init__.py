from .image import ImageTester
from .quant import QuantTester


def get_tester(tester_key):
    return {
        'image': ImageTester,
        'quant': QuantTester,
    }[tester_key]
