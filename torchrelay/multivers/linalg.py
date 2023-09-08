"""
Common linear algebra operations.

This module implements (almost) the same API as in `torch >= 2.0`.
It wraps torch functions when they are available, and implements
fallbacks for older versions.

# Overview

+----------------------------------------------------------+-------------------------------------------------------+
|**Matrix Properties**                                                                                             |
+----------------------------------------------------------+-------------------------------------------------------+
| [`norm`][torchrelay.multivers.linalg.norm]               | Computes a vector or matrix norm.                     |
+----------------------------------------------------------+-------------------------------------------------------+
| [`vector_norm`][torchrelay.multivers.linalg.vector_norm] | Computes a vector norm.                               |
+----------------------------------------------------------+-------------------------------------------------------+
| [`matrix_norm`][torchrelay.multivers.linalg.matrix_norm] | Computes a matrix norm.                               |
+----------------------------------------------------------+-------------------------------------------------------+
| [`diagonal`][torchrelay.multivers.linalg.diagonal]       | Alias for `torch.diagonal` with defaults              |
|                                                          | dim1= -2, dim2= -1.                                   |
+----------------------------------------------------------+-------------------------------------------------------+
| [`det`][torchrelay.multivers.linalg.det]                 | Computes the determinant of a square matrix.          |
+----------------------------------------------------------+-------------------------------------------------------+
| [`slogdet`][torchrelay.multivers.linalg.slogdet]         | Computes the sign and natural logarithm of the        |
|                                                          | absolute value of the determinant of a square matrix. |
+----------------------------------------------------------+-------------------------------------------------------+
| [`cond`][torchrelay.multivers.linalg.cond]               | Computes the condition number of a matrix with        |
|                                                          | respect to a matrix norm.                             |
+----------------------------------------------------------+-------------------------------------------------------+
| [`matrix_rank`][torchrelay.multivers.linalg.matrix_rank] | Computes the numerical rank of a matrix.              |
+----------------------------------------------------------+-------------------------------------------------------+
| **Decompositions**                                                                                               |
+----------------------------------------------------------+-------------------------------------------------------+
| [`cholesky`][torchrelay.multivers.linalg.cholesky]       | Computes the Cholesky decomposition of a complex      |
|                                                          | Hermitian or real symmetric positive-definite matrix. |
+----------------------------------------------------------+-------------------------------------------------------+
| [`qr`][torchrelay.multivers.linalg.qr]                   | Computes the QR decomposition of a matrix.            |
+----------------------------------------------------------+-------------------------------------------------------+
| [`lu`][torchrelay.multivers.linalg.lu]                   | Computes the LU decomposition with partial pivoting   |
|                                                          | of a matrix.                                          |
+----------------------------------------------------------+-------------------------------------------------------+
| [`lu_factor`][torchrelay.multivers.linalg.lu_factor]     | Computes a compact representation of the LU           |
|                                                          | factorization with partial pivoting of a matrix.      |
+----------------------------------------------------------+-------------------------------------------------------+
| [`eig`][torchrelay.multivers.linalg.eig]                 | Computes the eigenvalue decomposition of a square     |
|                                                          | matrix if it exists.                                  |
+----------------------------------------------------------+-------------------------------------------------------+
| [`eigvals`][torchrelay.multivers.linalg.eigvals]         | Computes the eigenvalues of a square matrix           |
+----------------------------------------------------------+-------------------------------------------------------+
| [`eigh`][torchrelay.multivers.linalg.eigh]               | Computes the eigenvalue decomposition of a complex    |
|                                                          | Hermitian or real symmetric matrix.                   |
+----------------------------------------------------------+-------------------------------------------------------+
| [`eigvalsh`][torchrelay.multivers.linalg.eigvalsh]       | Computes the eigenvalues of a complex Hermitian or    |
|                                                          | real symmetric matrix.                                |
+----------------------------------------------------------+-------------------------------------------------------+
| [`svd`][torchrelay.multivers.linalg.svd]                 | Computes the singular value decomposition (SVD) of    |
|                                                          | a matrix.                                             |
+----------------------------------------------------------+-------------------------------------------------------+
| [`svdvals`][torchrelay.multivers.linalg.svdvals]         | Computes the singular values of a matrix.             |
+----------------------------------------------------------+-------------------------------------------------------+
| **Solvers**                                                                                                      |
+----------------------------------------------------------+-------------------------------------------------------+
| [`solve`][torchrelay.multivers.linalg.solve]             | Computes the solution of a square system of linear    |
|                                                          | equations with a unique solution.                     |
+----------------------------------------------------------+-------------------------------------------------------+
| [`solve_triangular`]                                     | Computes the solution of a triangular system of       |
| [torchrelay.multivers.linalg.solve_triangular]           | linear equations with a unique solution..             |
+----------------------------------------------------------+-------------------------------------------------------+
| [`lu_solve`]                                             | Computes the solution of a square system of linear    |
| [torchrelay.multivers.linalg.lu_solve]                   | equations with a unique solution given an LU          |
|                                                          | decomposition.                                        |
+----------------------------------------------------------+-------------------------------------------------------+
| [`lstsq`]                                                | Computes a solution to the least squares problem of a |
| [torchrelay.multivers.linalg.lstsq]                      | system of linear equations.                           |
+----------------------------------------------------------+-------------------------------------------------------+
| **Inverses**                                                                                                     |
+----------------------------------------------------------+-------------------------------------------------------+
| [`inv`][torchrelay.multivers.linalg.inv]                 | Computes the inverse of a matrix.                     |
+----------------------------------------------------------+-------------------------------------------------------+
| [`pinv`][torchrelay.multivers.linalg.pinv]               | Computes the pseudoinverse (Moore-Penrose inverse) of |
|                                                          | a matrix.                                             |
+----------------------------------------------------------+-------------------------------------------------------+
| **Matrix Functions**                                                                                             |
+----------------------------------------------------------+-------------------------------------------------------+
| [`matrix_exp`][torchrelay.multivers.linalg.matrix_exp]   | Computes the matrix exponential of a square matrix.   |
+----------------------------------------------------------+-------------------------------------------------------+
| [`matrix_power`]                                         | Computes the n-th power of a square matrix for an     |
| [torchrelay.multivers.linalg.matrix_power]               | integer n.                                            |
+----------------------------------------------------------+-------------------------------------------------------+
| **Matrix Products**                                                                                              |
+----------------------------------------------------------+-------------------------------------------------------+
| [`cross`][torchrelay.multivers.linalg.cross]             | Computes the cross product of two 3-dimensional       |
|                                                          | vectors.                                              |
+----------------------------------------------------------+-------------------------------------------------------+
| [`matmul`][torchrelay.multivers.linalg.matmul]           | Matrix product of two tensors.                        |
+----------------------------------------------------------+-------------------------------------------------------+
| [`multi_dot`][torchrelay.multivers.linalg.multi_dot]     | Efficiently multiplies two or more matrices by        |
|                                                          | reordering the multiplications so that the fewest     |
|                                                          | arithmetic operations are performed..                 |
+----------------------------------------------------------+-------------------------------------------------------+
| **Misc**                                                                                                         |
+----------------------------------------------------------+-------------------------------------------------------+
| [`vander`][torchrelay.multivers.linalg.vander]           | Generates a Vandermonde matrix.                       |
+----------------------------------------------------------+-------------------------------------------------------+

"""  # noqa: E501
__all__ = [
    # matrix properties
    'norm',
    'vector_norm',
    'matrix_norm',
    'diagonal',
    'det',
    'slogdet',
    'cond',
    'matrix_rank',
    # decompositions
    'cholesky',
    'qr',
    'lu',
    'lu_factor',
    'eig',
    'eigvals',
    'eigh',
    'eigvalsh',
    'svd',
    'svdvals',
    # solvers
    'solve',
    'solve_triangular',
    'lu_solve',
    'lstsq',
    # inverses
    'inv',
    'pinv',
]

import torch
from collections import namedtuple
from functools import wraps
from .._torch_version import torch_version
from .._pyutils import ensure_list
from ._matrix_exp import matrix_exp_fallback, matrix_power_fallback
from .base import movedim, squeeze, unsqueeze, adjoint, conj


# ----------------------------------------------------------------------
#   Matrix properties
# ----------------------------------------------------------------------

if not hasattr(torch, 'linalg') and hasattr(torch.linalg, 'norm'):
    @wraps(torch.linalg.norm)
    def norm(
        A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None
    ):
        return torch.linalg.norm(
            A, ord, dim, keepdim, out=out, dtype=dtype
        )
else:
    @wraps(torch.norm)
    def norm(
        A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None
    ):
        return torch.norm(
            A, ord, dim, keepdim, out=out, dtype=dtype
        )


def vector_norm_fallback(
        x, ord=2, dim=None, keepdim=False, *, out=None, dtype=None
):
    r"""Computes a vector norm.

    Supports input of `float`, `double`, `cfloat` and `cdouble` dtypes.

    `ord` defines the norm that is computed. The following norms are supported:

    |  `ord`  |          vector norm          |
    | ------- | ----------------------------- |
    |  `inf`  | `max(abs(x))`                 |
    | `-inf`  | `min(abs(x))`                 |
    |  `0`    | `sum(x != 0)`                 |
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
            performing the operation, and the returned tensor's type
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
        if x.requires_grad:
            del opt['out']
            x = x.abs().reciprocal().sum(**opt).reciprocal()
            x = out.copy_(x)
        else:
            x = x.abs().reciprocal().sum(**opt).reciprocal_()
    elif ord == 2 and not x.is_complex():
        if x.requires_grad:
            del opt['out']
            x = x.square().sum(**opt).sqrt()
            x = out.copy_(x)
        else:
            x = x.square().sum(**opt).sqrt_()
    elif ord == -2 and not x.is_complex():
        if x.requires_grad:
            del opt['out']
            x = x.square().reciprocal().sum(**opt).sqrt().reciprocal()
            x = out.copy_(x)
        else:
            x = x.square().reciprocal().sum(**opt).sqrt_().reciprocal_()
    else:
        if x.requires_grad:
            del opt['out']
            x = x.abs().pow(ord).sum(**opt).pow(1/ord)
            x = out.copy_(x)
        else:
            x = x.abs().pow(ord).sum(**opt).pow_(1/ord)
    return x


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'vector_norm'):
    @wraps(torch.linalg.vector_norm)
    def vector_norm(
        x, ord=2, dim=None, keepdim=False, *, out=None, dtype=None
    ):
        return torch.linalg.vector_norm(
            x, ord, dim, keepdim, out=out, dtype=dtype
        )
else:
    @wraps(vector_norm_fallback)
    def vector_norm(
        x, ord=2, dim=None, keepdim=False, *, out=None, dtype=None
    ):
        return vector_norm_fallback(
            x, ord, dim, keepdim, out=out, dtype=dtype
        )


def matrix_norm_fallback(A, ord='fro', dim=(-2, -1), keepdim=False,
                         *, out=None, dtype=None):
    """Computes a matrix norm.

    Supports input of `float`, `double`, `cfloat` and `cdouble` dtypes.

    `ord` defines the norm that is computed. The following norms are supported:

    |  `ord`  |          matrix norm          |
    | ------- | ----------------------------- |
    | `'fro'` | `norm(eigvals(x), 2)`         |
    | `'nuc'` | `norm(eigvals(x), 1)`         |
    |  `inf`  | `max(sum(abs(x), dim=1))`     |
    | `-inf`  | `min(sum(abs(x), dim=1))`     |
    |  `1`    | `max(sum(abs(x), dim=0))`     |
    | `-1`    | `min(sum(abs(x), dim=0))`     |
    |  `2`    | `norm(eigvals(x), inf)`       |
    | `-2`    | `norm(eigvals(x), -inf)`      |

    Parameters
    ----------
    A : tensor
        tensor of shape `(*, n)` where `*` is zero or more batch dimensions.
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
        performing the operation, and the returned tensor's type
        will be dtype.

    Returns
    -------
    out : tensor
        A real-valued tensor, even when x is complex.

    """  # noqa: E501
    dim = ensure_list(dim)
    if len(dim) != 2:
        raise ValueError('`dim` should be a list of two integers')
    opt = dict(keepdim=keepdim, dim=dim, out=out, dtype=dtype)
    maxopt = dict(keepdim=keepdim, dim=dim, out=(out, None))

    if ord == 'fro':
        if A.requires_grad:
            del opt['out']
            A = A.square().sum(**opt).sqrt()
            return out.copy_(A)
        else:
            return A.square().sum(**opt).sqrt_()

    elif ord in (float('inf'), -float('inf'), 1, -1):
        opt[keepdim] = True
        opt[dim] = dim[0] if abs(ord) == 1 else dim[1]
        del opt['out']
        A = A.abs().sum(**opt)

        maxopt[keepdim] = True
        maxopt[dim] = dim[1] if abs(ord) == 1 else dim[0]
        A = A.max(**maxopt).values if ord > 0 else A.min(**maxopt).values

        if not keepdim:
            A = squeeze(A, dim)
        return A

    else:
        if ord == 'nuc':
            ord = 1
        elif ord == 2:
            ord = float('inf')
        elif ord == -2:
            ord = float('inf')
        else:
            raise ValueError(f'Unsupported matix norm `{ord}`')

        if out is not None:
            out = squeeze(out, dim)

        A = movedim(A, dim, (-2, -1))
        A = vector_norm(eigvals(A), ord, dim=-1, out=out)
        if keepdim:
            A = unsqueeze(A, dim)
        return A


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'matrix_norm'):
    @wraps(torch.linalg.matrix_norm)
    def matrix_norm(
        A, ord='fro', dim=(-2, -1), keepdim=False, *, out=None, dtype=None
    ):
        return torch.linalg.matrix_norm(
            A, ord, dim, keepdim, out=out, dtype=dtype
        )
else:
    @wraps(matrix_norm_fallback)
    def matrix_norm(
        A, ord='fro', dim=(-2, -1), keepdim=False, *, out=None, dtype=None
    ):
        return matrix_norm_fallback(
            A, ord, dim, keepdim, out=out, dtype=dtype
        )

if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'diagonal'):
    @wraps(torch.linalg.diagonal)
    def diagonal(A, *, offset=0, dim1=-2, dim2=-1):
        return torch.linalg.diagonal(A, offset=offset, dim1=dim1, dim2=dim2)
else:
    @wraps(torch.diagonal)
    def diagonal(A, *, offset=0, dim1=-2, dim2=-1):
        return torch.diagonal(A, offset, dim1, dim2)


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'det'):
    @wraps(torch.linalg.det)
    def det(A, *, out=None):
        return torch.linalg.det(A, out=out)
else:
    @wraps(torch.det)
    def det(A, *, out=None):
        return torch.det(A, out=out)


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'slogdet'):
    @wraps(torch.linalg.slogdet)
    def slogdet(A, *, out=None):
        return torch.linalg.slogdet(A, out=out)
else:
    @wraps(torch.slogdet)
    def slogdet(A, *, out=None):
        return torch.slogdet(A, out=out)


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'cond'):
    @wraps(torch.linalg.cond)
    def cond(A, ord=None, *, out=None):
        return torch.linalg.cond(A, ord, out=out)
elif hasattr(torch, 'cond'):
    @wraps(torch.cond)
    def cond(A, ord=None, *, out=None):
        return torch.cond(A, ord, out=out)
else:
    def cond(A, ord=None, *, out=None):
        if ord in (-2, 2):
            vals = svdvals(A)
            maxval = vals.max(-1).values
            minval = vals.min(-1).values
            c = maxval / minval if ord == 2 else minval / maxval
        else:
            c = matrix_norm(A, ord) * matrix_norm(A.inverse(), ord)
        return out.copy_(c) if out is not None else c


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'matrix_rank'):
    if torch_version('>=', (1, 11)):
        @wraps(torch.linalg.matrix_rank)
        def matrix_rank(A, *, atol=None, rtol=None, hermitian=False, out=None):
            return torch.linalg.matrix_rank(
                A, atol=atol, rtol=rtol, hermitian=hermitian, out=out
            )
    else:
        @wraps(torch.linalg.matrix_rank)
        def matrix_rank(A, *, atol=None, rtol=None, hermitian=False, out=None):
            return torch.linalg.matrix_rank(
                A, tol=atol, hermitian=hermitian, out=out
            )
else:
    def matrix_rank(A, *, atol=None, rtol=None, hermitian=False, out=None):
        rank = torch.matrix_rank(A, tol=atol, symmetric=hermitian)
        return out.copy_(rank) if out is not None else rank


# ----------------------------------------------------------------------
#   Decompositions
# ----------------------------------------------------------------------


def cholesky_fallback(A, *, upper=False, out=None):
    if out is not None:
        out = adjoint(out)
    C = torch.cholesky(A, upper=upper, out=out)
    if upper:
        C = adjoint(C)
        if out is not None:
            out = adjoint(out)
            out = out.copy_(C)
    return C


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'cholesky'):
    @wraps(torch.linalg.cholesky)
    def cholesky(A, *, upper=False, out=None):
        if upper and torch_version('<', (1, 10)):
            return cholesky_fallback(A, upper=upper, out=out)
        else:
            return torch.linalg.cholesky(A, upper=upper, out=out)
else:
    @wraps(cholesky_fallback)
    def cholesky(A, *, upper=False, out=None):
        return cholesky_fallback(A, upper=upper, out=out)


def qr_fallback(A, mode='reduced', *, out=None):
    some = mode.startwith('r')
    return torch.qr(A, some=some, out=out)


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'qr'):
    @wraps(torch.linalg.qr)
    def qr(A, mode='reduced', *, out=None):
        return torch.linalg.qr(A, mode=mode, out=out)
else:
    @wraps(qr_fallback)
    def qr(A, mode='reduced', *, out=None):
        return qr_fallback(A, mode=mode, out=out)


def lu_fallback(A, *, pivot=True, out=None):
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


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'lu'):
    @wraps(torch.linalg.lu)
    def lu(A, *, pivot=True, out=None):
        return torch.linalg.lu(A, pivot=pivot, out=out)
else:
    @wraps(lu_fallback)
    def lu(A, *, pivot=True, out=None):
        return lu_fallback(A, pivot=pivot, out=out)


def lu_factor_fallback(A, *, pivot=True, out=None):
    LU, P = torch.lu(A, pivot=pivot, out=out)
    return namedtuple('LU', 'LU pivots')(LU, P)


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'lu_factor'):
    @wraps(torch.linalg.lu_factor)
    def lu_factor(A, *, pivot=True, out=None):
        return torch.linalg.lu_factor(A, pivot=pivot, out=out)
else:
    @wraps(lu_factor_fallback)
    def lu_factor(A, *, pivot=True, out=None):
        return lu_factor_fallback(A, pivot=pivot, out=out)


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'eig'):
    @wraps(torch.linalg.eig)
    def eig(A, *, out=None):
        return torch.linalg.eig(A, out=out)
else:
    @wraps(torch.eig)
    def eig(A, *, out=None):
        return torch.eig(A, eigenvectors=True, out=out)


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'eigvals'):
    @wraps(torch.linalg.eigvals)
    def eigvals(A, *, out=None):
        return torch.linalg.eigvals(A, out=out)
else:
    @wraps(torch.eig)
    def eigvals(A, *, out=None):
        val, _ = torch.eig(A, out=(out, None))
        return val


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'eigh'):
    @wraps(torch.linalg.eigh)
    def eigh(A, UPLO='L', *, out=None):
        return torch.linalg.eigh(A, UPLO, out=out)
else:
    @wraps(torch.symeig)
    def eigh(A, UPLO='L', *, out=None):
        return torch.symeig(A, eigenvectors=True, upper=(UPLO == 'U'), out=out)


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'eigvalsh'):
    @wraps(torch.linalg.eigvalsh)
    def eigvalsh(A, UPLO='L', *, out=None):
        return torch.linalg.eigvalsh(A, UPLO, out=out)
else:
    @wraps(torch.symeig)
    def eigvalsh(A, UPLO='L', *, out=None):
        val, _ = torch.symeig(A, upper=(UPLO == 'U'), out=(out, None))
        return val


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'svd'):
    @wraps(torch.linalg.svd)
    def svd(A, full_matrices=True, *, out=None):
        return torch.linalg.svd(A, full_matrices, out=out)
else:
    @wraps(torch.svd)
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
    @wraps(torch.linalg.svdvals)
    def svdvals(A, *, out=None):
        return torch.linalg.svdvals(A, out=out)
else:
    @wraps(torch.svd)
    def svdvals(A, *, out=None):
        if out is not None:
            out = (None, out, None)
        _, S, _ = torch.svd(A, compute_uv=False, out=out)
        return S


# ----------------------------------------------------------------------
#   Solvers
# ----------------------------------------------------------------------

def solve_noleft_fallback(A, B, *, left=True, out=None):
    if left:
        return torch.linalg.solve(A, B, out=out)

    is_vector = B.ndim in (1, A.ndim - 1)
    if is_vector:
        B = B.unsqueeze(-1)
        if out is not None:
            out = out.unsqueeze(-1)
    if not left:
        A = A.transpose(-1, -2)
        B = B.transpose(-1, -2)
        if out is not None:
            out = out.transpose(-1, -2)

    out = torch.linalg.solve(A, B, out=out)

    if not left:
        out = out.transpose(-1, -2)
    if is_vector:
        out = out.squeeze(-1)
    return out


def solve_fallback(A, B, *, left=True, out=None):
    if B.ndim > 2 and B.ndim not in (A.ndim, A.ndim-1):
        raise RuntimeError("Incompatible matrix sizes for linalg_solve")

    is_vector = B.ndim in (1, A.ndim-1)
    if is_vector:
        B = B.unsqueeze(-1)
        if out is not None:
            out = out.unsqueeze(-1)
    if not left:
        A = A.transpose(-1, -2)
        B = B.transpose(-1, -2)
        if out is not None:
            out = out.transpose(-1, -2)

    out = torch.solve(B, A, out=(out, None))[0]

    if not left:
        out = out.transpose(-1, -2)
    if is_vector:
        out = out.squeeze(-1)
    return out


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'solve'):
    if torch_version('>=', (1, 13)):
        @wraps(torch.linalg.solve)
        def solve(A, B, *, left=True, out=None):
            return torch.linalg.solve(A, B, left=left, out=out)
    else:
        @wraps(solve_noleft_fallback)
        def solve(A, B, *, left=True, out=None):
            return solve_noleft_fallback(A, B, left=left, out=out)
else:
    @wraps(solve_fallback)
    def solve(A, B, *, left=True, out=None):
        return solve_fallback(A, B, left=left, out=out)


def solve_triangular_fallback(
    A, B, *, upper, left=True, unitriangular=False, out=None
):
    opt = dict(upper=upper, unitriangular=unitriangular)

    if not left:
        A = A.transpose(-1, -2)
        if out is not None:
            out = out.transpose(-1, -2)
        opt['transpose'] = True

    out = torch.solve(B, A, **opt, out=(out, None))[0]

    if not left:
        out = out.transpose(-1, -2)
    return out


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'solve_triangular'):
    @wraps(torch.linalg.solve_triangular)
    def solve_triangular(
        A, B, *, upper, left=True, unitriangular=False, out=None
    ):
        return torch.linalg.solve_triangular(
            A, B, upper=upper, left=left, unitriangular=unitriangular, out=out
        )
else:
    @wraps(solve_triangular_fallback)
    def solve_triangular(
            A, B, *, upper, left=True, unitriangular=False, out=None
    ):
        return solve_triangular_fallback(
            A, B, upper=upper, left=left, unitriangular=unitriangular, out=out
        )


def lu_solve_fallback(
    LU, pivots, B, *, left=True, adjoint=False, out=None
):
    if left and not adjoint:
        B = B.transpose(-1, -2)
        if out is not None:
            out = out.transpose(-1, -2)
    elif left and adjoint:
        B = adjoint(B)
        if out is not None:
            out = adjoint(out)
    elif adjoint:
        B = conj(B)
        if out is not None:
            out = conj(out)

    out = torch.lu_solve(B, LU, pivots)

    if left and not adjoint:
        out = out.transpose(-1, -2)
    elif left and adjoint:
        out = adjoint(out)
    elif adjoint:
        out = conj(out)

    return torch.lu_solve()


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'lu_solve'):
    @wraps(torch.linalg.lu_solve)
    def lu_solve(
        LU, pivots, B, *, left=True, adjoint=False, out=None
    ):
        return torch.linalg.solve_triangular(
            LU, pivots, B, left=left, adjoint=adjoint, out=out
        )
else:
    @wraps(lu_solve_fallback)
    def lu_solve(
        LU, pivots, B, *, left=True, adjoint=False, out=None
    ):
        return lu_solve_fallback(
            LU, pivots, B, left=left, adjoint=adjoint, out=out
        )


def lstsq_fallback(A, B, rcond=None):
    if A.ndim == B.ndim == 2:
        X = torch.lstsq(B, A).solution[:A.shape[1]]
    elif A.shape[-2] >= A.shape[-1]:
        AA = adjoint(A).matmul(A)
        AB = adjoint(A).matmul(B)
        X = torch.solve(AB, AA).solution
    else:
        U, S, V = torch.svd(A, some=True)
        X = adjoint(U).matmul(B)
        X = X / S.unsqueeze(-1)
        X = V.matmul(X)
    return X, None, None, None


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'lstsq'):
    @wraps(torch.linalg.lstsq)
    def lstsq(A, B, rcond=None):
        return torch.linalg.lstsq(A, B, rcond=rcond)
else:
    @wraps(lstsq_fallback)
    def lstsq(A, B, rcond=None):
        return lstsq_fallback(A, B, rcond=rcond)

if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'inv'):
    @wraps(torch.linalg.inv)
    def inv(A, *, out=None):
        return torch.linalg.inv(A, out=out)
else:
    @wraps(torch.inverse)
    def inv(A, *, out=None):
        return torch.inverse(A, out=out)

if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'pinv'):
    @wraps(torch.linalg.pinv)
    def pinv(A, *, atol=None, rtol=None, hermitian=False, out=None):
        return torch.linalg.pinv(
            A, atol=atol, rtol=rtol, hermitian=hermitian, out=out
        )
else:
    @wraps(torch.pinverse)
    def pinv(A, *, atol=None, rtol=None, hermitian=False, out=None):
        X = torch.pinverse(A, rcond=atol)
        if out is not None:
            out.copy_(X)
        return X

if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'matrix_exp'):
    @wraps(torch.linalg.matrix_exp)
    def matrix_exp(A):
        return torch.linalg.matrix_exp(A)
else:
    @wraps(matrix_exp_fallback)
    def matrix_exp(A):
        return matrix_exp_fallback(A)

if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'matrix_power'):
    @wraps(torch.linalg.matrix_power)
    def matrix_power(A, n, *, out=None):
        return torch.linalg.matrix_power(A, n, out=out)
else:
    @wraps(matrix_power_fallback)
    def matrix_power(A, n, *, out=None):
        P = matrix_power_fallback(A, n)
        if out is not None:
            out.copy_(P)
        return P

if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'cross'):
    @wraps(torch.linalg.cross)
    def cross(input, other, *, dim=-1, out=None):
        return torch.linalg.cross(input, other, dim=dim, out=out)
else:
    @wraps(torch.cross)
    def cross(input, other, *, dim=-1, out=None):
        return torch.cross(input, other, dim=dim, out=out)

if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'matmul'):
    @wraps(torch.linalg.matmul)
    def matmul(input, other, *, out=None):
        return torch.linalg.matmul(input, other, out=out)
else:
    @wraps(torch.matmul)
    def matmul(input, other, *, out=None):
        return torch.matmul(input, other, out=out)


def vecdot_fallback(input, other, *, dim=-1, out=None):
    input, other = torch.broadcast_tensors(input, other)
    input = movedim(input, dim, -1).unsqueeze(-2)
    other = movedim(other, dim, -1).unsqueeze(-1)
    if out is not None:
        out = out.unsqueeze(-1).unsqueeze(-1)
    out = matmul(input, other, out=out).squeeze(-1).squeeze(-1)
    return out


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'vecdot'):
    @wraps(torch.linalg.vecdot)
    def vecdot(input, other, *, dim=-1, out=None):
        return torch.linalg.vecdot(input, other, dim=dim, out=out)
else:
    @wraps(vecdot_fallback)
    def vecdot(input, other, *, out=None):
        return vecdot_fallback(input, other, out=out)


def multi_dot_order(tensors):
    # ported from
    # https://github.com/pytorch/pytorch/blob/
    # c55cb29bb205e94bb94cc3073e8b7da02af86430/
    # aten/src/ATen/native/LinearAlgebra.cpp#L850
    n = len(tensors)
    p = [t.shape[0] for t in tensors] + tensors[-1].shape[-1]
    m = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]
    for ell in range(1, n):
        for i in range(n-1):
            j = i + ell
            q0, s0 = float('inf'), None
            for k in range(i, j):
                q = m[i][k] + m[k+1][j] + p[i] * p[k+1] * p[j+1]
                if q < q0:
                    q0, s0 = q, k
            m[i][j], s[i][j] = q0, s0
    return s


def multi_dot_chain(tensors, order, i, j, out=None):
    if i == j:
        return tensors[i]
    else:
        return matmul(
            multi_dot_chain(tensors, order, i, order[i][j]),
            multi_dot_chain(tensors, order, order[i][j] + 1, j),
            out=out,
        )


def multi_dot_fallback(tensors, *, out=None):
    tensors = list(tensors)
    n = len(tensors)

    if n == 0:
        raise ValueError('Expected at least one tensor')
    if n == 1:
        if out is not None:
            out.copy_(tensors[0])
        return tensors[0]
    if len(tensors) == 2:
        return matmul(*tensors, out=out)

    vec_first = tensors[0].ndim == 1
    vec_last = tensors[-1].ndim == 1
    if vec_first:
        tensors[0] = tensors[0].unsqueeze(0)
    if vec_last:
        tensors[-1] = tensors[-1].unsqueeze(-1)
    if out is not None:
        if vec_first:
            out = out.unsqueeze(0)
        if vec_last:
            out = out.unsqueeze(-1)

    for t in tensors:
        if t.ndim != 2:
            raise ValueError('All tensors should be 2D')

    if len(tensors) == 3:
        a, b = tensors[0].shape
        c, d = tensors[2].shape
        cost_1 = (a * c) * (b + d)
        cost_2 = (b * d) * (a + c)
        if cost_1 > cost_2:
            out = matmul(tensors[0], matmul(tensors[1], tensors[2]), out=out)
        else:
            out = matmul(matmul(tensors[0], tensors[1]), tensors[2], out=out)
    else:
        order = multi_dot_order(tensors)
        out = multi_dot_chain(tensors, order, 0, n-1, out)

    if vec_first:
        out = out.squeeze(0)
    if vec_last:
        out = out.squeeze(-1)
    return out


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'multi_dot'):
    @wraps(torch.linalg.multi_dot)
    def multi_dot(tensors, *, out=None):
        return torch.linalg.multi_dot(tensors, out=out)
elif hasattr(torch, 'chain_matmul'):
    @wraps(torch.chain_matmul)
    def multi_dot(tensors, *, out=None):
        tensors = list(tensors)
        vec_first = tensors[0].ndim == 1
        vec_last = tensors[-1].ndim == 1
        if vec_first:
            tensors[0] = tensors[0].unsqueeze(0)
        if vec_last:
            tensors[-1] = tensors[-1].unsqueeze(-1)
        if out is not None:
            if vec_first:
                out = out.unsqueeze(0)
            if vec_last:
                out = out.unsqueeze(-1)
        if torch_version('>=', (1, 9)):
            out = torch.chain_matmul(*tensors, out=out)
        else:
            tmp = torch.chain_matmul(*tensors)
            if out is not None:
                out.copy_(tmp)
            out = tmp
        if vec_first:
            out = out.squeeze(0)
        if vec_last:
            out = out.squeeze(-1)
        return out
else:
    @wraps(multi_dot_fallback)
    def multi_dot(tensors, *, out=None):
        return multi_dot_fallback(tensors, out=out)


def vander_fallback(x, *, N=None):
    if N is None:
        N = x.shape[-1]
    pow = torch.arange(N, dtype=x.dtype, device=x.device)
    return x.unsqueeze(-1).pow(pow)


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'vander'):
    @wraps(torch.linalg.multi_dot)
    def vander(x, *, N=None):
        return torch.linalg.vander(x, N=N)
elif hasattr(torch, 'vander'):
    @wraps(torch.vander)
    def vander(x, *, N=None):
        return torch.vander(x, N=N)
else:
    @wraps(vander_fallback)
    def vander(x, *, N=None):
        return vander_fallback(x, N=N)
