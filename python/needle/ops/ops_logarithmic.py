from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

## TODO: Modify for better simplicity
class LogSoftmax(TensorOp):
    def compute(self, Z):
        Z -= Z.max(-1, keepdims=True)
        return Z - array_api.log(array_api.exp(Z).sum(-1, keepdims=True))

    def gradient(self, out_grad, node):
        Z = node.inputs[0]
        softmax_grad = exp(logsoftmax(Z))  # softmax(Z)
        return out_grad - softmax_grad * array_api.sum(out_grad.realize_cached_data(), axis=-1, keepdims=True)


def logsoftmax(a):
    return LogSoftmax()(a)


# Note: `axes` - Tuple of axes to sum and take the maximum element over. 
# This uses the same conventions as `needle.ops.Summation()`
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        self.max_Z = max_Z
        max_Z_reduced = array_api.max(Z, axis=self.axes)

        diff = Z - max_Z # implicit broadcasting
        e = array_api.exp(diff)
        se = array_api.sum(e, axis=self.axes) # don't keepdims
        lse = array_api.log(se)
        return lse + max_Z_reduced # prevent implicit broadcasting
    
    def gradient(self, out_grad, node):
        Z = node.inputs[0]
        # node.gradient
        
        e = exp(Z - self.max_Z)
        se = summation(e, axes=self.axes)
        lse = log(se)
        grad = Log().gradient(out_grad, lse)
        grad = Summation(axes=self.axes).gradient(grad, se)
        grad = Exp().gradient(grad, e)

        # print(Z.shape)
        # max_Z = array_api.max(Z, axis=self.axes, keepdims=True)

        # print(exp(Z).shape, summation(exp(Z), axes=self.axes).shape)
        # aa = exp(Z) / summation(exp(Z), axes=self.axes)



        return grad


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

