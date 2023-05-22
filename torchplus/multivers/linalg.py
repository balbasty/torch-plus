import torch
from collections import namedtuple
from torchplus import torch_version
from torchplus._pyutils import ensure_list
from torchplus.shapes import movedims


# ----------------------------------------------------------------------
#   Matrix properties
# ----------------------------------------------------------------------


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'norm'):
    norm = torch.linalg.norm
else:
    norm = torch.norm


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'vector_norm'):
    vector_norm = torch.linalg.vector_norm
else:
    def vector_norm(x, ord=2, dim=None, keepdim=False, *, out=None, dtype=None):
        """Computes a vector norm.

        Supports input of `float`, `double`, `cfloat` and `cdouble` dtypes.

        `ord` defines the norm that is computed. The following norms are supported:

        | ======= | ============================= |
        |   ord   |          vector norm          |
        | ======= | ============================= |
        | `inf`   | `max(abs(x))`                 |
        | `-inf`  | `min(abs(x))`                 |
        | `0`     | `sum(x != 0)`                 |
        | other   | `sum(abs(x)**ord)**(1./ord)`  |

        Parameters
        ----------
        x : tensor
            tensor of shape `(*, n)` where *` is zero or more batch dimensions.
        ord : {int, float, inf, -inf}
            The order of norm.
        dim :  int or tuple[int]
            Dimensions over which to compute the vector norm.
        keepdim : bool
            If set to True, the reduced dimensions are retained in the
            result as dimensions with size one.

        Other Parameters
        ----------------
        out : tensor
            The output tensor. Ignored if None.
        dtype : torch.dtype
             If specified, the input tensor is cast to dtype before
             performing the operation, and the returned tensor’s type
             will be dtype.

        Returns
        -------
        out : tensor
            A real-valued tensor, even when x is complex.

        """
        if dim is None:
            dim = list(range(x.ndim))
        dim = ensure_list(dim)
        opt = dict(keepdim=keepdim, dim=dim, out=out, dtype=dtype)
        maxopt = dict(keepdim=keepdim, dim=dim, out=(out, None))

        if ord in ('fro', 'nuc'):
            raise ValueError(f'Unsupported norm "{ord}" for vectors.')
        if ord == float('inf'):
            x = x.abs().max(**maxopt).values
        elif ord == -float('inf'):
            x = x.abs().min(**maxopt).values
        elif ord == 0:
            x = (x != 0).sum(**opt)
        elif ord == 1:
            x = x.abs().sum(**opt)
        elif ord == -1:
            x = x.abs().reciprocal_().sum(**opt).reciprocal_()
        elif ord == 2 and not x.is_complex():
            x = x.square().sum(**opt).sqrt_()
        elif ord == -2 and not x.is_complex():
            x = x.square().reciprocal_().sum(**opt).sqrt_().reciprocal_()
        else:
            x = x.abs()
            x = x.pow_(ord).sum(**opt).pow_(1/ord)
        return x


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'matrix_norm'):
    matrix_norm = torch.linalg.matrix_norm
else:
    def matrix_norm(A, ord='fro', dim=(-2, -1), keepdim=False,
                    *, out=None, dtype=None):
        return torch.norm(A, ord, dim, keepdim, out=out, dtype=dtype)


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'diagonal'):
    diagonal = torch.linalg.diagonal
else:
    def diagonal(A, *, offset=0, dim1=-2, dim2=-1):
        """Alias for torch.diagonal() with defaults `dim1=-2`, `dim2=-1.`"""
        return torch.diagonal(A, offset, dim1, dim2)


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'slogdet'):
    slogdet = torch.linalg.slogdet
else:
    slogdet = torch.slogdet


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'cond'):
    cond = torch.linalg.cond
if hasattr(torch, 'cond'):
    cond = torch.cond
else:
    def cond(A, ord=None, *, out=None):
        if ord in (-2, 2):
            vals = eigvals(A)
            maxval = vals.max(-1).values
            minval = vals.min(-1).values
            c = maxval / minval if ord == 2 else minval / maxval
        else:
            c = matrix_norm(A, ord) * matrix_norm(A.inverse(), ord)
        return out.copy_(c) if out is not None else c


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'matrix_rank'):
    matrix_rank = torch.linalg.matrix_rank
else:
    def matrix_rank(A, *, atol=None, rtol=None, hermitian=False, out=None):
        rank = torch.matrix_rank(A, tol=atol, symmetric=hermitian)
        return out.copy_(rank) if out is not None else rank


# ----------------------------------------------------------------------
#   Decompositions
# ----------------------------------------------------------------------


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'cholesky'):
    cholesky = torch.linalg.cholesky
else:
    cholesky = torch.cholesky


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'qr'):
    qr = torch.linalg.qr
else:
    qr = torch.qr


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'lu'):
    lu = torch.linalg.lu
else:
    def lu(A, *, pivot=True, out=None):
        LU, P = torch.lu(A, pivot=pivot)
        P, L, U = torch.lu_unpack(LU, P)
        if out is not None:
            if out[0] is not None:
                out[0].copy_(P)
            if out[1] is not None:
                out[1].copy_(L)
            if out[2] is not None:
                out[2].copy_(U)
        return namedtuple('PLU', 'P L U')(P, L, U)


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'lu_factor'):
    lu_factor = torch.linalg.lu_factor
else:
    def lu_factor(A, *, pivot=True, out=None):
        LU, P = torch.lu(A, pivot=pivot, out=out)
        return namedtuple('LU', 'LU pivots')(LU, P)


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'eig'):
    eig = torch.linalg.eig
else:
    def eig(A, *, out=None):
        return torch.eig(A, eigenvectors=True, out=out)


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'eigvals'):
    eigvals = torch.linalg.eigvals
else:
    def eigvals(A, *, out=None):
        val, _ = torch.eig(A, out=(out, None))
        return val


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'eigh'):
    eigh = torch.linalg.eigh
else:
    def eigh(A, UPLO='L', *, out=None):
        return torch.symeig(A, eigenvectors=True, upper=(UPLO == 'U'), out=out)


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'eigvalsh'):
    eigvalsh = torch.linalg.eigvalsh
else:
    def eigvalsh(A, UPLO='L', *, out=None):
        val, _ = torch.symeig(A, upper=(UPLO == 'U'), out=(out, None))
        return val


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'svd'):
    svd = torch.linalg.svd
else:
    def svd(A, full_matrices=True, *, out=None):
        if out is not None:
            out = list(out)
            out[-1] = out[-1].transpose(-1, -2)
        U, S, V = torch.svd(A, some=not full_matrices, out=out)
        Vh = V.transpose(-1, -2).conj()
        if out is not None:
            out[-1] = out[-1].transpose(-1, -2)
            out[-1].copy_(Vh)
        return U, S, Vh


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'svdvals'):
    svdvals = torch.linalg.svdvals
else:
    def svdvals(A, *, out=None):
        if out is not None:
            out = (None, out, None)
        _, S, _ = torch.svd(A, compute_uv=False, out=out)
        return S
