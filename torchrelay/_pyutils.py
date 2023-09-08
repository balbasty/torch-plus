"""Internal python utilities (not part of the API)"""
from types import GeneratorType as generator


def move_to_permutation(length, source, destination):

    source = ensure_list(source)
    destination = ensure_list(destination)
    if len(destination) == 1:
        # we assume that the user wishes to keep moved dimensions
        # in the order they were provided
        destination = destination[0]
        if destination >= 0:
            destination = list(range(destination, destination+len(source)))
        else:
            destination = list(range(destination+1-len(source), destination+1))
    if len(source) != len(destination):
        raise ValueError('Expected as many source as destination positions.')
    source = [length + src if src < 0 else src for src in source]
    destination = [length + dst if dst < 0 else dst for dst in destination]
    if len(set(source)) != len(source):
        raise ValueError(f'Expected source positions to be unique but got '
                         f'{source}')
    if len(set(destination)) != len(destination):
        raise ValueError(f'Expected destination positions to be unique but got '
                         f'{destination}')

    # compute permutation
    positions_in = list(range(length))
    positions_out = [None] * length
    for src, dst in zip(source, destination):
        positions_out[dst] = src
        positions_in[src] = None
    positions_in = filter(lambda x: x is not None, positions_in)
    for i, pos in enumerate(positions_out):
        if pos is None:
            positions_out[i], *positions_in = positions_in

    return positions_out


def ensure_list(x, dim=None):
    """Ensure that an object is a list (of size at last dim)

    If x is a list, nothing is done (no copy triggered).
    If it is a tuple, it is converted into a list.
    Otherwise, it is placed inside a list.

    This function is less versatile (but much faster) than `make_list`.
    """
    if not isinstance(x, (list, tuple, generator, range)):
        x = [x]
    x = list(x)
    if dim and len(x) < dim:
        x += x[-1:] * (dim - len(x))
    return x


def prod(sequence, inplace=False):
    """Perform the product of a sequence of elements.

    Parameters
    ----------
    sequence : any object that implements `__iter__`
        Sequence of elements for which the `__mul__` operator is defined.
    inplace : bool, default=False
        Perform the product inplace (using `__imul__` instead of `__mul__`).

    Returns
    -------
    product :
        Product of the elements in the sequence.

    """
    accumulate = None
    for elem in sequence:
        if accumulate is None:
            accumulate = elem
        elif inplace:
            accumulate *= elem
        else:
            accumulate = accumulate * elem
    return accumulate