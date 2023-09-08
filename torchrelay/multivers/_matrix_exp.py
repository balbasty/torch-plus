"""
Implementation of the matrix exponential and its derivative.

This implementation is based on John Ashburner's in SPM, which relies
on a Taylor approximation for both the exponential and its derivative.
Faster implementations that rely on scaling and squaring or Pade
approximations (as in scipy) could be used instead. This may be the
object of future work.
"""
import torch
from torch.nn import functional as F
from typing import Tuple
import math

try:
    from torch.cuda.amp import custom_fwd, custom_bwd

except ImportError:

    def _identity(x):
        return x

    def custom_fwd(*a, **k):
        return a[0] if a and callable(a[0]) else _identity

    def custom_bwd(*a, **k):
        return a[0] if a and callable(a[0]) else _identity


def matrix_exp_fallback(X):
    return _ExpM.apply(X)


def matrix_power_fallback(X, n):
    if n < 0:
        X = X.inverse()
        n = -n
    return _PowM.apply(X, n)


@torch.jit.script
def matrix_exp_forward(X, max_order: int = 10000, tol: float = 1e-32):
    dim = X.shape[-1]
    I = torch.eye(dim, dtype=X.dtype, device=X.device)   # noqa: E741
    En = I + X   # expm(X)
    Xn = X       # X**n / n!
    for n_order in range(2, max_order+1):
        Xn = torch.matmul(Xn, X)
        Xn = Xn / n_order
        En = En + Xn
        if tol > 0:
            sos = torch.dot(Xn.flatten(), Xn.flatten())
            if sos <= torch.numel(Xn) * tol:
                break
    return En


@torch.jit.script
def matrix_exp_backward(
    X, max_order: int = 10000, tol: float = 1e-32
) -> Tuple[torch.Tensor, torch.Tensor]:

    dim = X.shape[-1]
    dX = torch.arange(dim ** 2, dtype=torch.long, device=X.device)
    dX = F.one_hot(dX).to(dtype=X.dtype, device=X.device)
    dX = dX.reshape((dim, dim, dim, dim))
    X = X[..., None, None, :, :]

    # At this point:
    #   X.shape        = [*batch, 1, 1, D, D]
    #   dX.shape       = [*batch, D, D, D, D]

    I = torch.eye(dim, dtype=X.dtype, device=X.device)   # noqa: E741
    En = I + X   # expm(X)
    Xn = X       # X**n / n!
    dEn = dX     # dexpm(X)/dX
    dXn = dX     # X**n / dX / n!

    for n_order in range(2, max_order+1):
        # Compute coefficients at order `n_order`, and accumulate
        dXn = torch.matmul(dXn, X) + torch.matmul(Xn, dX)
        dXn = dXn / n_order
        dEn = dEn + dXn
        Xn = torch.matmul(Xn, X)
        Xn = Xn / n_order
        En = En + Xn
        # Compute sum-of-squares
        if tol > 0:
            sos = torch.dot(Xn.flatten(), Xn.flatten())
            if sos <= torch.numel(Xn) * tol:
                break

    En = En[..., 0, 0, :, :]
    return En, dEn


class _ExpM(torch.autograd.Function):
    """Matrix exponential with automatic differentiation."""

    @staticmethod
    @custom_fwd
    def forward(ctx, X):
        if X.requires_grad:
            E, G = matrix_exp_backward(X)
            ctx.save_for_backward(G)
        else:
            E = matrix_exp_forward(X)
        return E

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):
        grad, = ctx.saved_tensors
        grad *= output_grad[..., None, None, :, :]
        grad = grad.sum(dim=[-1, -2])
        return grad


@torch.jit.script
def matrix_power_square(X, n: int):
    """Compute X**(2**n) by recursive squaring"""
    for _ in range(n):
        X = X.matmul(X)
    return X


@torch.jit.script
def matrix_power_forward(X, n: int):
    """Compute X**n, using recursive squaring when possible"""
    if n == 0:
        dim = X.shape[-1]
        I = torch.eye(dim, dtype=X.dtype, device=X.device)   # noqa: E741
        return I.expand(X.shape)
    if n == 1:
        return X
    m = max(0, int(math.floor(math.log2(n))))
    Xn = matrix_power_square(X, m)
    n = n - 2**m
    while n:
        if n == 1:
            Xn = Xn.matmul(X)
            n -= 1
        else:
            m = max(0, int(math.floor(math.log2(n))))
            Xn = Xn.matmul(matrix_power_square(X, m))
            n = n - 2**m
    return Xn


@torch.jit.script
def matrix_power_backward(X, n: int) -> Tuple[torch.Tensor, torch.Tensor]:

    if n == 0:
        dim = X.shape[-1]
        I = torch.eye(dim, dtype=X.dtype, device=X.device)   # noqa: E741
        return I.expand(X.shape), torch.zeros_like(X)

    dim = X.shape[-1]
    dX = torch.arange(dim ** 2, dtype=torch.long, device=X.device)
    dX = F.one_hot(dX).to(dtype=X.dtype, device=X.device)
    dX = dX.reshape((dim, dim, dim, dim))
    X = X[..., None, None, :, :]

    # At this point:
    #   X.shape        = [*batch, 1, 1, D, D]
    #   dX.shape       = [*batch, D, D, D, D]

    Xn = X       # X**n
    dXn = dX     # X**n / dX
    for _ in range(2, n+1):
        dXn = torch.matmul(dXn, X) + torch.matmul(Xn, dX)
        Xn = torch.matmul(Xn, X)

    Xn = Xn[..., 0, 0, :, :]
    return Xn, dXn


class _PowM(torch.autograd.Function):
    """Matrix power with automatic differentiation."""

    @staticmethod
    @custom_fwd
    def forward(ctx, X, n):
        if X.requires_grad:
            E, G = matrix_power_backward(X, n)
            ctx.save_for_backward(G)
        else:
            E = matrix_power_forward(X, n)
        return E

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):
        grad, = ctx.saved_tensors
        grad *= output_grad[..., None, None, :, :]
        grad = grad.sum(dim=[-1, -2])
        return grad, None
