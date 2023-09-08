"""
# Overview

+-------------------------------------------------+------------------------------------+
| **Modules**                                                                          |
+-------------------------------------------------+------------------------------------+
| [`bounds`][torchrelay.extra.bounds]             | Boundary conditions.               |
+-------------------------------------------------+------------------------------------+
| **Padding**                                                                          |
+-------------------------------------------------+------------------------------------+
| [`pad`][torchrelay.extra.pad]                   | Pad a tensor.                      |
+-------------------------------------------------+------------------------------------+
| [`ensure_shape`][torchrelay.extra.ensure_shape] | Pad/crop a tensor so               |
|                                                 | that it has a given shape.         |
+-------------------------------------------------+------------------------------------+
| [`roll`][torchrelay.extra.roll]                 | Like `torch.roll`, but with        |
|                                                 | any boundary condition.            |
+-------------------------------------------------+------------------------------------+
| **Labels**                                                                           |
+-------------------------------------------------+------------------------------------+
| [`isin`][torchrelay.extra.isin]                 | Returns a mask for elements        |
|                                                 | that belong to labels.             |
+-------------------------------------------------+------------------------------------+
| [`one_hot`][torchrelay.extra.one_hot]           | One-hot encode a volume of labels. |
+-------------------------------------------------+------------------------------------+
| [`relabel`][torchrelay.extra.relabel]           | Relabel a label tensor according   |
|                                                 | to a lookup table.                 |
+-------------------------------------------------+------------------------------------+

"""  # noqa: E501
__all__ = [
    'pad',
    'ensure_shape',
    'roll',
    'isin',
    'relabel',
    'one_hot',
    'bounds',
]

from ._pad import pad, ensure_shape
from ._roll import roll
from ._labels import isin, relabel, one_hot
from . import bounds
