import sys


def is_valid_cuda_version():
    import torch.cuda as cuda
    if cuda.is_available():
        from torch.version import __version__
        return not __version__.startswith("2.1")
    return False


def isMacOS():
    return sys.platform.startswith('darwin')