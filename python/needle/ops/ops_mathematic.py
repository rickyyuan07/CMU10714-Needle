"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a ** b
        
    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        ret1 = out_grad * rhs * power(lhs, rhs - 1)
        ret2 = out_grad * log(lhs) * power(lhs, rhs)
        return ret1, ret2

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        return out_grad * self.scalar * array_api.power(node.inputs[0], self.scalar - 1)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / (rhs * rhs)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


# Note: the definition of Transpose and np.transpose are different
class Transpose(TensorOp): # reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, `axes` - tuple)
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            # Reverse the last two axes
            axe = tuple(range(len(a.shape) - 2)) + (len(a.shape) - 1, len(a.shape) - 2)
        else:
            # Reverse the given two axes, self.axes = (axis1, axis2)
            axe = list(range(len(a.shape)))
            axe[self.axes[0]], axe[self.axes[1]] = axe[self.axes[1]], axe[self.axes[0]]
        return a.permute(axe)
        

    def gradient(self, out_grad, node):
        # reverse the axes again
        return transpose(out_grad, axes=self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp): # gives a new shape to an array without changing its data (1 input, `shape` - tuple)
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp): # broadcast an array to a new shape (1 input, `shape` - tuple)
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        out_shape = out_grad.shape

        # Pad input shape with ones on the left if necessary
        # Note: By default, BroadcastTo aligns to the right
        # e.g. (4,)->(4,5) is not valid since alignment 4 and 5 not match; (4,1)->(4,5) is valid
        if len(input_shape) < len(out_shape):
            input_shape = (1,) * (len(out_shape) - len(input_shape)) + input_shape

        # Calculate the axes along which broadcasting happened
        axes = []
        for i, (dim_out, dim_in) in enumerate(zip(out_shape, input_shape)):
            if dim_in == 1 and dim_out > 1:
                axes.append(i)

        # Sum the gradient along the broadcasted dimensions
        grad = summation(out_grad, axes=tuple(axes))
        # Reshape for unsqueezing, e.g. (3,) -> (3, 1)
        grad = reshape(grad, node.inputs[0].shape)
        return grad


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp): # sum of array elements over given axes (1 input, `axes` - tuple)
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape

        # If axes are None (summation over all axes), treat it as reducing all axes
        if self.axes is None:
            axes = tuple(range(len(input_shape)))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = self.axes

        # Compute the target shape by adding 1's "in place" of the summed axes
        target_shape = list(input_shape)
        for axis in sorted(axes):
            target_shape[axis] = 1
        
        # Reshape the gradient to the target shape
        out_grad = out_grad.reshape(target_shape)
        return broadcast_to(out_grad, input_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        # Gradient with respect to lhs (left-hand side)
        lhs_grad = out_grad @ transpose(rhs)
        
        # Gradient with respect to rhs (right-hand side)
        rhs_grad = transpose(lhs) @ out_grad
        
        # Handle broadcasting for lhs gradient
        if len(lhs_grad.shape) > len(lhs.shape):
            # Reduce extra dimensions in lhs_grad to match lhs
            axes_to_sum = tuple(range(len(lhs_grad.shape) - len(lhs.shape)))
            lhs_grad = lhs_grad.sum(axes=axes_to_sum)
        
        if len(rhs_grad.shape) > len(rhs.shape):
            # Reduce extra dimensions in rhs_grad to match rhs
            axes_to_sum = tuple(range(len(rhs_grad.shape) - len(rhs.shape)))
            rhs_grad = rhs_grad.sum(axes=axes_to_sum)

        return lhs_grad, rhs_grad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp): # numerical negative, element-wise (1 input)
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return a.maximum(0)

    def gradient(self, out_grad, node):
        input_nparr = node.inputs[0].realize_cached_data()
        return out_grad * (input_nparr > 0)


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        # return out_grad * (1 - node ** 2)
        return out_grad * (-node ** 2 + 1)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        assert len(args) > 0, "Need at least one array to stack"
        shape = args[0].shape
        for arg in args:
            assert arg.shape == shape, "All arrays must have the same shape"
            assert arg.dtype == args[0].dtype, "All arrays must have the same dtype"
            assert arg.device == args[0].device, "All arrays must be on the same device"
        
        new_shape = list(shape)
        new_shape.insert(self.axis, len(args))
        # Create a new array with the new shape
        ret = array_api.empty(new_shape, dtype=args[0].dtype, device=args[0].device)

        # Fill the new array with the input arrays
        for i, arg in enumerate(args):
            # Create slice object to insert along the new axis
            slices = [slice(None)] * len(new_shape)  # slice(None) is equivalent to ':'
            slices[self.axis] = i
            ret[tuple(slices)] = arg  # ret[:, ..., i, :, ...] = arg
        return ret
            
    def gradient(self, out_grad, node):
        return split(out_grad, axis=self.axis)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        # Calculate the size of each split
        ret_shape = [dim for i, dim in enumerate(A.shape) if i != self.axis]

        ret = []
        for i in range(A.shape[self.axis]):
            slices = [slice(None)] * len(A.shape)
            slices[self.axis] = i
            # Need to call compact() before reshaping
            ret.append(A[tuple(slices)].compact().reshape(ret_shape))
        
        return tuple(ret)

    def gradient(self, out_grad, node):
        return stack(out_grad, axis=self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

    def compute(self, a):
        return array_api.flip(a, self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, axes=self.axes)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        if self.dilation <= 0:
            return a

        # Construct new shape with dilation
        new_shape = list(a.shape)
        for axis in self.axes:
            if axis >= len(a.shape):
                continue
            new_shape[axis] = new_shape[axis] * (self.dilation + 1)
        
        # Construct slices for dilation
        slices = [slice(None)] * len(new_shape)
        for axis in self.axes:
            if axis >= len(a.shape):
                continue
            slices[axis] = slice(None, None, self.dilation + 1)
        
        ret = array_api.full(new_shape, 0, dtype=a.dtype, device=a.device)
        ret[tuple(slices)] = a
        return ret

    def gradient(self, out_grad, node):
        return undilate(out_grad, axes=self.axes, dilation=self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        if self.dilation <= 0:
            return a

        # Construct slices for undilation
        slices = [slice(None)] * len(a.shape)
        for axis in self.axes:
            if axis >= len(a.shape):
                continue
            slices[axis] = slice(None, None, self.dilation + 1)
        
        return a[tuple(slices)]

    def gradient(self, out_grad, node):
        return dilate(out_grad, axes=self.axes, dilation=self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Permute(TensorOp):
    def __init__(self, axes: tuple):
        self.axes = axes

    def compute(self, a):
        return a.compact().permute(self.axes)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        index = [0] * len(self.axes)
        for i in range(len(self.axes)):
            index[self.axes[i]] = i
        return permute(out_grad, tuple(index))
        
def permute(a, axes):
    return Permute(axes)(a)

class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


