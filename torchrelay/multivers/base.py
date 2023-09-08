__all__ = [
    'movedim',
    'squeeze',
    'unsqueeze',
]

import torch
from functools import wraps
from .._pyutils import move_to_permutation, ensure_list
from .._torch_version import torch_version


def movedim1_fallback(x, source: int, destination: int):
    """Backward compatible `torch.movedim` for single dimensions"""
    dim = x.ndim
    src, dst = source, destination
    src = dim + src if src < 0 else src
    dst = dim + dst if dst < 0 else dst
    permutation = [d for d in range(dim)]
    permutation = permutation[:src] + permutation[src+1:]
    permutation = permutation[:dst] + [src] + permutation[dst:]
    return x.permute(permutation)


def movedim_fallback(input, source, destination):
    """Moves the position of one or more dimensions

    Other dimensions that are not explicitly moved remain in their
    original order and appear at the positions not specified in
    destination.

    History
    -------
    !!! added "1.7"
        `torch.movedim` was introduced in torch version `1.7`.

    Parameters
    ----------
    input : tensor
        Input tensor
    source : int or sequence[int]
        Original positions of the dims to move. These must be unique.
    destination : int or sequence[int]
        Destination positions for each of the original dims.
        These must also be unique.

    Returns
    -------
    output : tensor
        Tensor with moved dimensions.

    """
    if isinstance(source, int) and isinstance(destination, int):
        return movedim1_fallback(input, source, destination)
    else:
        perm = move_to_permutation(input.dim(), source, destination)
        return input.permute(*perm)


if hasattr(torch, 'movedim'):
    @wraps(torch.movedim)
    def movedim(input, source, destination):
        return torch.movedim(input, source, destination)
else:
    @wraps(movedim_fallback)
    def movedim(input, source, destination):
        return movedim_fallback(input, source, destination)


def squeeze_fallback(input, dim=None):
    r"""
    Returns a tensor with all specified dimensions of `input` of size 1 removed

    For example, if `input` is of shape:
    $(A \times 1 \times B \times C \times 1 \times D)$ then the
    `input.squeeze()` will be of shape $(A \times B \times C \times D)$.

    When `dim` is given, a squeeze operation is done only in the given
    dimension(s). If `input` is of shape: $(A \times 1 \times B)$,
    `input.squeeze(0)` leaves the tensor unchanged, but `input.squeeze(1)`
    will squeeze the tensor to the shape $(A \times B)$.

    !!! note
        The returned tensor shares the storage with the input tensor,
        so changing the contents of one will change the contents of the other.

    !!! note
        A dim value within the range `[-input.dim() - 1, input.dim() + 1)`
        can be used. Negative dim will correspond to `squeeze()` applied
        at `dim = dim + input.dim() + 1`.

    !!! warning
        If the tensor has a batch dimension of size 1, then `input.squeeze()`
        will also remove the batch dimension, which can lead to
        unexpected errors. Consider specifying only the dims you wish to
        be squeezed.

    History
    -------
    !!! added "2.0"
        Since torch version 2.0, `dim` accepts tuples of dimensions.

    Parameters
    ----------
    input : tensor
        The input tensor
    dim : int or tuple[int], optional
        If given the input will be squeezed only in the specified dimensions.

    Returns
    -------
    output : tensor
        Tensor with squeezed dimensions.
    """
    if dim is None or isinstance(dim, int):
        return torch.squeeze(input, dim)
    slicer = [slice(None)] * input.ndim
    for d in dim:
        if input.shape[d] == 1:
            slicer[d] = 0
    return input[tuple(slicer)]


if torch_version('>=', (2, 0)):
    @wraps(torch.squeeze)
    def squeeze(input, dim=None):
        return torch.squeeze(input, dim=dim)
else:
    @wraps(squeeze_fallback)
    def squeeze(input, dim=None):
        return squeeze_fallback(input, dim=dim)


def unsqueeze_fallback(input, dim):
    r"""
    Returns a new tensor with a dimension of size one inserted at the
    specified position.

    !!! note
        The returned tensor shares the storage with the input tensor,
        so changing the contents of one will change the contents of the other.

    !!! note
        A dim value within the range `[-input.dim() - 1, input.dim() + 1)`
        can be used. Negative dim will correspond to `unsqueeze()` applied
        at `dim = dim + input.dim() + 1`.

    !!! tip
        Contrary to the native `torch.unsqueeze`, this function accepts
        tuples of dimensions (like `torch.squeeze` since version 2.0)


    Parameters
    ----------
    input : tensor
        The input tensor
    dim : int or tuple[int]
        The index (indices) at which to insert the singleton dimension(s).

    Returns
    -------
    output : tensor
        Tensor with unsqueezed dimensions.
    """
    if isinstance(dim, int):
        return torch.unsqueeze(input, dim)
    dim = ensure_list(dim)
    slicer = [slice(None)] * (input.ndim + len(dim))
    for d in dim:
        slicer[d] = None
    return input[tuple(slicer)]


@wraps(squeeze_fallback)
def unsqueeze(input, dim):
    return unsqueeze_fallback(input, dim)


if hasattr(torch, 'conj'):
    # since 1.4
    @wraps(torch.conj)
    def conj(input):
        return torch.conj(input)
else:
    def conj(input):
        return input


def adjoint_fallback(x):
    return conj(x.transpose(-1, -2))


if hasattr(torch, 'adjoint'):
    # since 1.11
    @wraps(torch.adjoint)
    def adjoint(input):
        return torch.adjoint(input)
else:
    @wraps(adjoint_fallback)
    def adjoint(input):
        return adjoint_fallback(input)
