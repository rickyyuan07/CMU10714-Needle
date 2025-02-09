import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)


def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2) # gain for ReLU
    bound = gain * math.sqrt(3 / fan_in)
    if shape is None:
        ret = rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    else:
        ret = rand(*shape, low=-bound, high=bound, **kwargs)
    return ret


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2) # gain for ReLU
    std = gain / math.sqrt(fan_in)
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)
