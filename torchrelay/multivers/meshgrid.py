__all__ = [
    'meshgrid_ij',
    'meshgrid_xy',
]
r"""
# meshgrid

For `torch<1.10`, `torch.meshgrid` worked in `"ij"` mode, meaning that
the first tensor of coordinates was mapped to the first dimension of the
output grid and the second tensor of coordinates was mapped to the
second dimension of the output grid. Starting with `torch=1.10`,
the keyword argument `indexing`, which takes value `"ij"` or `"xy"`, was
introduced. Furthermore, the default behavior of the function when `indexing`
is not used will change from `"ij"` to `"xy"` in the future.

To make any code backward compatible, we define explicit functions
postfixed by either `_ij` or `_xy`.
"""

import torch
from torchrelay import torch_version


_help_intro = (
r"""Creates grids of coordinates specified by the 1D inputs in `tensors`.

This is helpful when you want to visualize data over some
range of inputs.

Given $N$ 1D tensors $T_0, \dots, T_{N-1}$ as inputs with
corresponding sizes $S_0, \dots, S_{N-1}$, this creates $N$
N-dimensional tensors $G_0, \dots, G_{N-1}$, each with shape
$(S_0, \dots, S_{N-1})$ where the output $G_i$ is constructed
by expanding $T_i$ to the result shape.

!!! note
    0D inputs are treated equivalently to 1D inputs of a
    single element.
""")  # noqa: E122

_help_prm = (
r"""
Parameters
----------
*tensors : tensor
    list of scalars or 1 dimensional tensors. Scalars will be
    treated as tensors of size $(1,)$ automatically

Returns
-------
seq : list[tensor]
    list of expanded tensors

""")  # noqa: E122

_help_warnxy = (
r"""
!!! warning
    In mode `xy`, the first dimension of the output corresponds to the
    cardinality of the second input and the second dimension of the output
    corresponds to the cardinality of the first input.
""")  # noqa: E122

_help_ij = _help_intro + _help_prm
_help_xy = _help_intro + _help_warnxy + _help_prm


if torch_version('>=', (1, 10)):
    # torch >= 1.10
    # -> use `indexing` keyword

    def meshgrid_ij(*tensors):
        return list(torch.meshgrid(*tensors, indexing='ij'))

    def meshgrid_xy(*tensors):
        return list(torch.meshgrid(*tensors, indexing='xy'))


else:
    # torch < 1.10
    # -> implement "xy" mode manually

    def meshgrid_ij(*tensors):
        return list(torch.meshgrid(*tensors))

    def meshgrid_xy(*tensors):
        grid = list(torch.meshgrid(*tensors))
        if len(grid) > 1:
            grid[0] = grid[0].transpose(0, 1)
            grid[1] = grid[1].transpose(0, 1)
        return grid


meshgrid_ij.__doc__ = _help_ij
meshgrid_xy.__doc__ = _help_xy
