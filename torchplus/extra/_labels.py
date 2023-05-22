import torch
from torchplus._pyutils import ensure_list
from torchplus.multivers import movedim


def isin(tensor, labels):
    """Returns a mask for elements that belong to labels

    Parameters
    ----------
    tensor : (*shape_tensor) tensor_like
        Input tensor
    labels : (*shape_labels, nb_labels) tensor_like
        Labels.
        `shape_labels` and `shape_tensor` should be broadcastable.

    Returns
    -------
    mask : (*shape) tensor[bool]

    """

    tensor = torch.as_tensor(tensor)
    if isinstance(labels, set):
        labels = list(labels)
    labels = torch.as_tensor(labels)

    if labels.shape[-1] == 1:
        # only one label in the list
        return tensor == labels[..., 0]

    mask = tensor.new_zeros(tensor.shape, dtype=torch.bool)
    for label in torch.unbind(labels, dim=-1):
        mask = mask | (tensor == label)

    return mask


def one_hot(x, dim=-1, exclude_labels=None, exclude_missing=False, max_label=None,
            implicit=False, implicit_index=0, dtype=None, return_lookup=False):
    """One-hot encode a volume of labels.

    !!! note
        This function extends `torch.nn.functional.one_hot`.

    Parameters
    ----------
    x : tensor
        An integer-type tensor with label values.
    dim : int, default=-1
        Dimension in which to insert the one-hot channel.
    exclude_labels : sequence[int], optional
        A list of labels to exclude from one-hot encoding.
    exclude_missing : bool, default=False
        Exclude missing labels from one-hot encoding
        (their channel will be squeezed)
    max_label : int, optional
        Maximum label value
    implicit : bool, default=False
        Make the returned tensor have an implicit background class.
        In this case, output probabilities do not sum to one, but to some
        value smaller than one.
    implicit_index : int, default=-1
        Output channel to make implicit
    dtype : tensor.dtype, optional
        Output data type.
    return_lookup : bool, default=False
        Return lookup table from one-hot indices to labels

    Returns
    -------
    y : tensor
        One-hot tensor.
        The number of one-hot channels is equal to `x.max() - len(exclude) + 1`
        if not `implicit` else `x.max() - len(exclude)`.

    """
    nb_classes = (max_label or int(x.max().item())) + 1
    exclude_labels = set(ensure_list(exclude_labels or []))
    if exclude_missing:
        all_labels = x.unique()
        missing_labels = [i for i in range(nb_classes) if i not in all_labels]
        exclude_labels = exclude_labels.union(missing_labels)

    dtype = dtype or x.dtype
    out = torch.zeros([nb_classes-implicit, *x.shape], dtype=dtype, device=x.device)
    implicit_index = (nb_classes + implicit_index if implicit_index < 0 else
                      implicit_index)
    i = 0
    lookup = []
    for j in range(nb_classes):
        if j in exclude_labels:
            continue
        if i == implicit_index:
            implicit_index = None
            continue
        out[i] = (x == j)
        lookup.append(j)
        i += 1

    out = movedim(out, 0, dim)
    return (out, lookup) if return_lookup else out


def merge_labels(x, lookup):
    """Relabel a label tensor according to a lookup table

    Parameters
    ----------
    x : tensor
    lookup : sequence of [sequence of] int

    Returns
    -------
    x : tensor

    """
    out = torch.zeros_like(x)
    for i, j in enumerate(lookup):
        j = ensure_list(j)
        out[isin(x, j)] = i
    return out