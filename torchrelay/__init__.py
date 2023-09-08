"""
# Overview

+---------------------------------------------------------+------------------------------------------------------------------------+
| **Modules**                                                                                                                      |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`extra`][torchrelay.extra]                             | Extended variants of functions from the PyTorch functional API.        |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`itertools`][torchrelay.itertools]                     | Same as the itertools python package, but in PyTorch.                  |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`multivers`][torchrelay.multivers]                     | Backward-compatible PyTorch API.                                       |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`jit`][torchrelay.jit]                                 | TorchScript utilities.                                                 |
+---------------------------------------------------------+------------------------------------------------------------------------+
| **Backends**                                                                                                                     |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`to`][torchrelay.to]                                   | Move/convert to a common dtype or device.                              |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`to_max_backend`][torchrelay.to_max_backend]           | Move to a common dtype or device.                                      |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`to_max_device`][torchrelay.to_max_device]             | Move to a common device.                                               |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`to_max_dtype`][torchrelay.to_max_dtype]               | Move to a common dtype.                                                |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`get_backend`][torchrelay.get_backend]                 | Return the backend (dtype and device) of a tensor.                     |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`max_backend`][torchrelay.max_backend]                 | Get the (max) dtype and device.                                        |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`max_device`][torchrelay.max_device]                   | Find a common device for all inputs.                                   |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`max_dtype`][torchrelay.max_dtype]                     | Find the maximum data type from a series of inputs.                    |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`as_torch_dtype`][torchrelay.as_torch_dtype]           | Convert a numpy data type (or a data type name) to a torch data dtype. |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`as_numpy_dtype`][torchrelay.as_numpy_dtype]           | Convert a torch data type (or a data type name) to a numpy data dtype. |
+---------------------------------------------------------+------------------------------------------------------------------------+
| **Indexing**                                                                                                                     |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`moveelem`][torchrelay.moveelem]                       | Move elements in a tensor.                                             |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`slice_tensor_along`][torchrelay.slice_tensor_along]   | Index a tensor along one dimensions.                                   |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`slice_tensor`][torchrelay.slice_tensor]               | Index a tensor along one or several dimensions.                        |
+---------------------------------------------------------+------------------------------------------------------------------------+
| **Shapes**                                                                                                                       |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`movedims`][torchrelay.movedims]                       | Moves the position of one or more dimensions.                          |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`movedims_front2back`][torchrelay.movedims_front2back] | Move the first N dimensions to the back.                               |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`movedims_back2front`][torchrelay.movedims_back2front] | Move the last N dimensions to the front.                               |
+---------------------------------------------------------+------------------------------------------------------------------------+
| [`shiftdim`][torchrelay.movedims_back2front]            | Shift the dimensions of x by n.                                        |
+---------------------------------------------------------+------------------------------------------------------------------------+

"""  # noqa: E501
__all__ = [
    'torch_version'
]

from . import _version
from ._torch_version import torch_version  # noqa: F401

from . import backends      # noqa: F401 device/dtype
from . import jit           # noqa: F401 just-in-time utilities
from . import multivers     # noqa: F401 multi-version compatibility
from . import indexing      # noqa: F401 indexing utilities
from . import extra         # noqa: F401 pytorch extensions
from . import itertools     # noqa: F401 implements `itertools` for tensors
from . import shapes        # noqa: F401 shapes/strides modifiers

from .backends import *     # noqa: F401, F403
from .indexing import *     # noqa: F401, F403
from .shapes import *       # noqa: F401, F403

__version__ = _version.get_versions()['version']

__all__ += backends.__all__
__all__ += indexing.__all__
__all__ += shapes.__all__
