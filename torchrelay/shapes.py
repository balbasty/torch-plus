__all__ = [
    'movedims',
    'movedims_front2back',
    'movedims_back2front',
    'shiftdim',
]
import torch
from ._pyutils import move_to_permutation


def movedims(input, source, destination):
    """Moves the position of one or more dimensions

    Other dimensions that are not explicitly moved remain in their
    original order and appear at the positions not specified in
    destination.

    Parameters
    ----------
    input : tensor
        Input tensor
    source : int or sequence[int]
        Original positions of the dims to move. These must be unique.
    destination : int or sequence[int]
        Destination positions for each of the original dims.
        These must also be unique.

        !!! tip "If a single destination is provided"
            - if it is negative, the last source dimension is moved to
              `destination` and all other source dimensions are moved to its
              left.
            - if it is positive, the first source dimension is moved to
              `destination` and all other source dimensions are moved to its
              right.

    Returns
    -------
    output : tensor
        Tensor with moved dimensions.

    """
    perm = move_to_permutation(input.dim(), source, destination)
    return input.permute(*perm)


def movedims_front2back(tensor, dim):
    """Move the first N dimensions to the back"""
    dims = list(range(tensor.dim()))
    perm = dims[dim:] + dims[:dim]
    return tensor.permute(*perm)


def movedims_back2front(tensor, dim):
    """Move the last N dimensions to the front"""
    dims = list(range(tensor.dim()))
    perm = dims[-dim:] + dims[:-dim]
    return tensor.permute(*perm)


def shiftdim(x, n=None):
    """Shift the dimensions of x by n.

    Parameters
    ----------
    x : tensor
        Input tensor.
    n : int, default=None
        Shift.

        * When `n` is positive, `shiftdim` shifts the dimensions to
          the left and wraps the N leading dimensions to the end.
        * When `n` is negative, `shiftdim` shifts the dimensions to
          the right and pads with singletons.
        * When `n` is None, `shiftdim` removes all leading singleton
          dimensions. The number of removed dimensions is returned
          as well.

    Returns
    -------
    x : tensor
        Output tensor.
    n : int, if n is None
        Number of removed dimensions

    """
    if n is None:
        shape = torch.as_tensor(x.size())
        n = (shape != 1).nonzero()
        if n.numel() == 0:
            n = x.dim()
            x = x.reshape([])
        else:
            n = n[0]
            x = x.reshape(shape[n:].tolist())
        return x, n
    elif n < 0:
        x = x.reshape((1,)*(-n) + x.size())
    elif n > 0:
        n = n % x.dim()
        x = x.permute(tuple(range(n, x.dim())) + tuple(range(n)))
    return x
