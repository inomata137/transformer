gpu = False

def gpu_config(_gpu: bool | None = None):
    '''configure whether to use cupy
    _gpu: Set True to use cupy, False to numpy.

    returns updated result
    '''
    global gpu
    if type(_gpu) is bool:
        gpu = _gpu
    return gpu
