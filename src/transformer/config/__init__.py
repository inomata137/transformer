_gpu = False

def gpu_config(gpu: bool | None = None):
    '''configure whether to use cupy
    gpu: Set True to use cupy, False to numpy.

    returns updated result
    '''
    global _gpu
    if type(gpu) is bool:
        _gpu = gpu
    return _gpu
