__all__ = [
    'to',
    'to_max_backend',
    'to_max_device',
    'to_max_dtype',
    'get_backend',
    'max_backend',
    'max_device',
    'max_dtype',
    'as_torch_dtype',
    'as_numpy_dtype',
]
import torch
import numbers
import numpy as np


def to(*args, dtype=None, device=None):
    """Move/convert to a common dtype or device.

    Parameters
    ----------
    *args : tensor_like
        Input tensors or tensor-like objects
    dtype : str or torch.dtype, optional
        Target data type
    device : str or torch.device, optional
        Target device

    Returns
    -------
    *args : tensor_like
        Converted tensors

    """
    if len(args) == 1:
        return torch.as_tensor(args[0], dtype=dtype, device=device)
    else:
        return tuple(torch.as_tensor(arg, dtype=dtype, device=device)
                     if arg is not None else arg for arg in args)


def to_max_backend(*args, force_float=False, dtype=None, device=None):
    """Move to a common dtype and device.

    See `max_dtype` and `max_device`.

    Parameters
    ----------
    *args : tensor_like
    force_float : bool, default=False

    Returns
    -------
    *args_to : tensor

    """
    if len(args) == 0:
        return
    dtype = max_dtype(*args, dtype, force_float=force_float)
    device = max_device(*args, device)
    if len(args) == 1:
        return torch.as_tensor(args[0], dtype=dtype, device=device)
    else:
        return tuple(torch.as_tensor(arg, dtype=dtype, device=device)
                     if arg is not None else arg for arg in args)


def to_max_device(*args):
    """Move to a common device.

    See `max_device`.

    Parameters
    ----------
    *args : tensor_like

    Returns
    -------
    *args_to : tensor

    """
    if len(args) == 0:
        return
    device = max_device(*args)
    if len(args) == 1:
        return torch.as_tensor(args[0], device=device)
    else:
        return tuple(torch.as_tensor(arg, device=device)
                     for arg in args)


def to_max_dtype(*args):
    """Move to a common data type.

    See `max_dtype`.

    Parameters
    ----------
    *args : tensor_like

    Returns
    -------
    *args_to : tensor

    """
    if len(args) == 0:
        return
    dtype = max_dtype(*args)
    if len(args) == 1:
        return torch.as_tensor(args[0], dtype=dtype)
    else:
        return tuple(torch.as_tensor(arg, dtype=dtype)
                     for arg in args)


def get_backend(x):
    """Return the backend (dtype and device) of a tensor

    Parameters
    ----------
    x : tensor

    Returns
    -------
    dict with keys 'dtype' and 'device'

    """
    return dict(dtype=x.dtype, device=x.device)


def max_backend(*args, dtype=None, device=None):
    """Get the (max) dtype and device.

    Parameters
    ----------
    args : tensors

    Returns
    -------
    dict with keys 'dtype' and 'device'

    """
    return dict(dtype=max_dtype(*args, dtype),
                device=max_device(*args, device))


def max_device(*args):
    """Find a common device for all inputs.

    If at least one input object is on a CUDA device:

    * if all cuda object are on the same cuda device, return it
    * if some objects are on different cuda devices, return
        `device('cuda')` without an index.

    Else, return device('cpu') or None.

    Parameters
    ----------
    *args : tensor_like or device_like

    Returns
    -------
    device : torch.device

    """
    is_tensor = torch.is_tensor
    def is_array(x): return isinstance(x, np.ndarray)

    def select_device(*many_devices):
        if len(many_devices) == 0:
            return None
        elif len(many_devices) == 1:
            return many_devices[0]
        device1, device2, *many_devices = many_devices
        if len(many_devices) > 0:
            return select_device(
                select_device(device1, device2), *many_devices)
        if device1 is None:
            return device2
        elif device2 is None:
            return device1
        elif device1.type == 'cuda' and device2.type != 'cuda':
            return device1
        elif device2.type == 'cuda' and device1.type != 'cuda':
            return device2
        elif device1.index is None:
            return device2
        elif device2.index is None:
            return device1
        elif device1.index == device2.index:
            return device1
        else:
            return torch.device('cuda')

    def explore_device(x):
        if x is None:
            return None
        if isinstance(x, (torch.device, str)):
            return torch.device(x)
        elif is_tensor(x):
            return x.device
        elif is_array(x) or isinstance(x, numbers.Number):
            # numpy/builtin type: None
            return None
        else:
            # assume it is a sequence: check what we find in there
            devices = [explore_device(elem) for elem in x]
            return select_device(*devices)

    return explore_device(args)


def max_dtype(*args, force_float=False):
    """Find the maximum data type from a series of inputs.

    The returned dtype is the best one to use for upcasting the objects.

    * Tensors and arrays have priority python objects.
    * Tensors and arrays with non-null dimensionality have priority
        over scalars.
    * If any of the torch/numpy objects have a floating point type
        a floating point type is returned.
    * If any of the objects is complex, a complex type is returned.
    * If all torch/numpy objects have an integer type and there is
        an integer type that avoids overflowing, it is returned.
    * If no integer type that ensures underflowing exists, the default
        floating point data type is returned.
    * If `force_float is True`, a floating point data type is returned
        even if all input objects have an integer data type.

    Parameters
    ----------
    *args : tensor_like or type_like
    force_float : bool, default=False

    Returns
    -------
    dtype : torch.dtype

    """
    is_tensor = torch.is_tensor

    def is_array(x):
        return isinstance(x, np.ndarray)

    def is_np_dtype(x):
        return isinstance(x, np.dtype) or \
               (isinstance(x, type) and issubclass(x, np.number))

    def is_torch_dtype(x):
        return isinstance(x, torch.dtype)

    def is_py_dtype(x):
        return isinstance(x, type) and issubclass(x, numbers.Number)

    def is_dtype(x):
        return is_torch_dtype(x) or is_np_dtype(x) or is_py_dtype(x)

    def upcast(*many_types):
        if len(many_types) == 0:
            return None
        elif len(many_types) == 1:
            return many_types[0]
        dtype1, dtype2, *many_types = many_types
        if len(many_types) > 0:
            return upcast(upcast(dtype1, dtype2), *many_types)
        # here, we only have torch dtypes
        if dtype1 is None:
            return dtype2
        elif dtype2 is None:
            return dtype1
        elif dtype1 is torch.complex128 or dtype2 is torch.complex128:
            return torch.complex128
        elif dtype1 is torch.complex64 or dtype2 is torch.complex64:
            return torch.complex64
        elif hasattr(torch, 'complex32') and (dtype1 is torch.complex32 or
                                              dtype2 is torch.complex32):
            return torch.complex32
        elif dtype1 is torch.float64 or dtype2 is torch.float64:
            return torch.float64
        elif dtype1 is torch.float32 or dtype2 is torch.float32:
            return torch.float32
        elif dtype1 is torch.float16 or dtype2 is torch.float16:
            return torch.float16
        elif dtype1 is torch.int64 or dtype2 is torch.int64:
            return torch.int64
        elif dtype1 is torch.int32 or dtype2 is torch.int32:
            return torch.int32
        elif dtype1 is torch.int16 or dtype2 is torch.int16:
            return torch.int16
        elif dtype1 is torch.int8 and dtype2 is torch.int8:
            return torch.int8
        elif dtype1 is torch.uint8 and dtype2 is torch.uint8:
            return torch.uint8
        elif (dtype1 is torch.int8 and dtype2 is torch.uint8) or \
             (dtype1 is torch.uint8 and dtype2 is torch.int8):
            return torch.int16
        elif dtype1 is torch.bool and dtype2 is torch.bool:
            return torch.bool
        else:
            raise TypeError('We do not deal with type {} or {} yet.'
                            .format(dtype1, dtype2))

    def explore_dtype(x, n_pass=1):
        # find the max data type at a given pass
        if x is None:
            return None
        elif is_dtype(x):
            return as_torch_dtype(x)
        elif (is_tensor(x) or is_array(x)) and len(x.shape) > 0:
            return as_torch_dtype(x.dtype)
        elif is_tensor(x) or is_array(x):
            # scalar: only return if pass 2+
            return as_torch_dtype(x.dtype) if n_pass >= 2 else None
        elif isinstance(x, numbers.Number):
            # builtin type:  only return if pass 3+
            return as_torch_dtype(type(x)) if n_pass >= 3 else None
        else:
            # assume it is a sequence: check what we find in there
            return upcast(*[explore_dtype(elem, n_pass) for elem in x])

    # 1) tensors/arrays with dim > 0
    maxdtype = explore_dtype(args, n_pass=1)

    # 2) tensor/arrays with dim == 0
    if maxdtype is None:
        maxdtype = upcast(maxdtype, explore_dtype(args, n_pass=2))

    # 3) tensor/arrays
    if maxdtype is None:
        maxdtype = upcast(maxdtype, explore_dtype(args, n_pass=3))

    # Finally) ensure float
    if force_float:
        maxdtype = upcast(maxdtype, torch.get_default_dtype())

    return maxdtype


def as_torch_dtype(dtype, byteswap=True, upcast=False):
    """Convert a numpy data type (or a data type name) to a torch data dtype.

    !!! warning
        Builtin Python data types `int` and `float` are mapped to
        `torch.int32` and `torch.float32`, to match `torch.as_tensor`'s
        behavior. It differs from `as_numpy_dtype`, which maps these
        types to `np.int64` and `np.float64`.

    Parameters
    ----------
    dtype : str or np.dtype or torch.dtype
        Input data type
    byteswap : bool
        If the data type is not implemented in PyTorch but its
        byteswapped version is, return the byteswapped version.
        If `False`, raise a `TypeError`.
    upcast : bool
        If the data type is not implemented in PyTorch, but its values
        can be represented by a larger data type that exists in PyTorch,
        return the larger data type.
        If `False`, raise a `TypeError`.

    Returns
    -------
    dtype : torch.dtype
        Torch data type

    """
    if dtype is int:
        return torch.int32
    if dtype is float:
        return torch.float32
    if dtype is complex:
        return torch.complex64
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype = np.dtype(dtype)
    if dtype.byteswap != '=' and not byteswap:
        raise TypeError('Only native byte orders are implemented in PyTorch. '
                        'Use `byteswap=True` to automatically byteswap dtype')

    if hasattr(torch, dtype.name):
        return getattr(torch, dtype.name)
    if not upcast:
        raise TypeError('Datatype', dtype, 'cannot be represented in PyTorch. '
                        'Use `upcast=True` to automatically upcast dtype')

    kind2name = {'u': 'uint', 'i': 'int', 'f': 'float', 'c': 'complex'}
    if dtype.kind not in kind2name:
        raise TypeError('Datatype', dtype, 'cannot be represented in PyTorch')

    basename = kind2name[dtype.kind]
    itemsize = dtype.itemsize
    while not hasattr(torch, f'{basename}{itemsize}'):
        itemsize *= 2
        if hasattr(torch, f'{basename}{itemsize}'):
            return getattr(torch, f'{basename}{itemsize}')
        if itemsize > 256:
            break
    raise TypeError('Datatype', dtype, 'cannot be represented in PyTorch')


def as_numpy_dtype(dtype, upcast=False):
    """Convert a torch data type (or a data type name) to a torch data dtype.

    !!! warning
        Builtin Python data types `int` and `float` are mapped to
        `np.int64` and `np.float64`, to match `np.asarray`'s
        behavior. It differs from `as_torch_dtype`, which maps these
        types to `np.int32` and `np.float32`.

    Parameters
    ----------
    dtype : str or np.dtype or torch.dtype
        Input data type
    upcast : bool
        If the data type is not implemented in NumPy, but its values
        can be represented by a larger data type that exists in NumPy,
        return the larger data type.
        If `False`, raise a `TypeError`.

    Returns
    -------
    dtype : np.dtype
        Numpy data type

    """
    if dtype in _torch_dtypes_to_name:
        dtype_name = _torch_dtypes_to_name[dtype]
        try:
            return np.dtype(dtype_name)
        except TypeError:
            pass
    else:
        return np.dtype(dtype)

    if not upcast:
        raise TypeError('Datatype', dtype, 'cannot be represented in NumPy. '
                        'Use `upcast=True` to automatically upcast dtype')

    kind2name = {'u': 'uint', 'i': 'int', 'f': 'float', 'c': 'complex'}
    basename = kind2name[dtype_name[0]]
    itemsize = ''
    while dtype_name.endswith(list('0123456789')):
        itemsize = dtype_name[-1] + itemsize
        dtype_name = dtype_name[:-1]
    itemsize = int(itemsize)
    while not hasattr(np, f'{basename}{itemsize}'):
        itemsize *= 2
        if hasattr(np, f'{basename}{itemsize}'):
            return np.dtype(getattr(np, f'{basename}{itemsize}'))
        if itemsize > 256:
            break
    raise TypeError('Datatype', dtype, 'cannot be represented in PyTorch')


_possible_torch_dtypes = (
    'bool',
    'uint8', 'uint16', 'uint32', 'uint64', 'uint256',
    'int8', 'int16', 'int32', 'int64', 'int256',
    'float16', 'float32', 'float64', 'float128',
    'complex32', 'complex64', 'complex128', 'complex256',
)
_torch_dtypes = {
    key: getattr(torch, key)
    for key in _possible_torch_dtypes
    if hasattr(torch, key)
}
_torch_dtypes_to_name = {
    getattr(torch, key): key
    for key in _possible_torch_dtypes
    if hasattr(torch, key)
}
