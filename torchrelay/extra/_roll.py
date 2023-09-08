import torch
from .._pyutils import ensure_list, prod
from ..jit import meshgrid_list_ij, sub2ind_list
from . import bounds


def roll(inp, shifts=1, dims=None, bound='circular'):
    r"""Like `torch.roll`, but with any boundary condition

    !!! warning
        When `dims` is `None`, we do not flatten but shift all dimensions.
        This differs from the behavior of `torch.roll` .

    Parameters
    ----------
    inp : tensor
        Input
    shifts : [sequence of] int
        Amount by which to roll.
        Positive shifts to the right, negative to the left.
    dims : [sequence of] int
        Dimensions to roll.
        By default, shifts apply to all dimensions if a scalar,
        or to the last N if a sequence.
    bound : "{'constant', 'replicate', 'reflect', 'mirror', 'circular'}"
        Boundary condition

    Returns
    -------
    out : tensor
        Rolled tensor

    """
    if dims is None:
        if isinstance(shifts, int):
            dims = list(range(inp.dim()))
        else:
            shifts = ensure_list(shifts)
            dims = list(range(-len(shifts), 0))
    dims = ensure_list(dims)
    shifts = ensure_list(shifts, len(dims))
    bound = map(bounds.to_nitorch, ensure_list(bound, len(dims)))
    bound = [getattr(bounds, b + '_') for b in bound]

    grid = [torch.arange(n, device=inp.device) for n in inp.shape]
    mult = [1] * inp.dim()
    for d, s, b in zip(dims, shifts, bound):
        grid[d] -= s
        grid[d], mult[d] = b(grid[d], inp.shape[d])
    grid = list(meshgrid_list_ij(grid))
    if any(map(torch.is_tensor, mult)):
        mult = meshgrid_list_ij(mult)
    mult = prod(mult)
    grid = sub2ind_list(grid, inp.shape)

    out = inp.flatten()[grid]
    out *= mult
    return out
