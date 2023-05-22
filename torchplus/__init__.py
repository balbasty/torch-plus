from . import _version
__version__ = _version.get_versions()['version']

from ._torch_version import torch_version


from . import jit           # just-in-time utilities
from . import multivers     # multi-version compatibility
from . import indexing      # indexing utilities
from . import extra         # pytorch extensions
from . import itertools     # implements `itertools` for tensors
from . import shapes        # shapes/strides modifiers (do not touch values)
