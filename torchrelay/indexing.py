__all__ = [
    'moveelem',
    'slice_tensor_along',
    'slice_tensor',
]
import torch
from torchrelay._pyutils import move_to_permutation, ensure_list


def moveelem(input, source, destination, dim=-1):
    """Move elements in a tensor

    Parameters
    ----------
    input : tensor
        Input tensor
    source : [sequence of] int
        Source indices of elements to move
    destination : [sequence of] int
        Target indices of moved elements
    dim : int, default=-1
        Dimension along which to move elements

    Returns
    -------
    output : tensor

    """
    perm = move_to_permutation(input.shape[dim], source, destination)
    perm = torch.as_tensor(perm, dtype=torch.long, device=input.device)
    return input.index_select(dim, perm)


def slice_tensor_along(x, index, dim=-1):
    """Index a tensor along one dimensions.

    This function is relatively similar to `torch.index_select`, except
    that it uses the native indexing mechanism and can therefore
    returns a tensor that use the same storage as the input tensor.

    It is faster but less versatile than `slice_tensor`.

    Parameters
    ----------
    x : tensor
        Input tensor.
    index : int or list[int] or slice
        Indices to select along `dim`.
    dim : int, default=last
        Dimension to index.

    Returns
    -------
    y : tensor
        Output tensor.

    """
    slicer = [slice(None)] * x.dim()
    slicer[dim] = index
    slicer = tuple(slicer)
    return x[slicer]


def slice_tensor(x, index, dim=None):
    """Index a tensor along one or several dimensions.

    This function is relatively similar to `torch.index_select`, except
    that it uses the native indexing mechanism and can therefore
    returns a tensor that use the same storage as the input tensor.

    Parameters
    ----------
    x : tensor
        Input tensor.
    index : index_like or tuple[index_like]
        Indices to select along each dimension in `dim`.
        If multiple dimensions are indexed, they *must* be held in a
        tuple (not a list). Each index can be a long, list of long,
        slice or tensor of long, but *cannot* be an ellipsis or
        tensor of bool.
    dim : int or sequence[int], optional
        Dimensions to index. If it is a list, `index` *must* be a tuple.
        By default, the last `n` dimensions (where `n` is the number of
        indices in `index`) are used.


    Returns
    -------
    y : tensor
        Output tensor.

    """
    # format (dim, index) as (list, tuple) with same length
    if not isinstance(index, tuple):
        index = (index,)
    if dim is None:
        dim = list(range(-len(index), 0))
    dim = ensure_list(dim)
    nb_dim = max(len(index), len(dim))
    dim = ensure_list(dim, nb_dim)
    index = ensure_list(index, nb_dim)

    # build index
    full_index = [slice(None)] * x.dim()
    for d, ind in zip(dim, index):
        if ind is Ellipsis or (torch.is_tensor(ind) and
                               ind.dtype == torch.bool):
            raise TypeError('`index` cannot be an ellipsis or mask')
        full_index[d] = ind
    full_index = tuple(full_index)

    return x.__getitem__(full_index)
