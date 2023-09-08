
def norm(
    A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None

):
    """Computes a vector or matrix norm

    `ord` defines the norm that is computed. The following norms are supported:

    |  `ord`  |          matrix norm          |          vector norm         |
    | ------- | ----------------------------- | ---------------------------- |
    | `None`  | Frobenius norm (see below)    | 2-norm (see below)           |
    | `'fro'` | `norm(eigvals(x), 2)`         | _not supported_              |
    | `'nuc'` | `norm(eigvals(x), 1)`         | _not supported_              |
    |  `inf`  | `max(sum(abs(x), dim=1))`     | `max(abs(x))`                |
    | `-inf`  | `min(sum(abs(x), dim=1))`     | `min(abs(x))`                |
    |  `0`    | _not supported_               | `sum(x != 0)`                |
    |  `1`    | `max(sum(abs(x), dim=0))`     | see below                    |
    | `-1`    | `min(sum(abs(x), dim=0))`     | see below                    |
    |  `2`    | `norm(eigvals(x), inf)`       | see below                    |
    | `-2`    | `norm(eigvals(x), -inf)`      | see below                    |
    | other   | _not supported_               | `sum(abs(x)**ord)**(1./ord)` |

    !!! note "Data types"

        Supports input of `float`, `double`, `cfloat` and `cdouble` dtypes.

        If `A` is complex valued, it computes the norm of `A.abs()`.

        `dtype` may be used to perform the computation in a more precise
        dtype. It is semantically equivalent to calling
        `linalg.norm(A.to(dtype))` but it is faster in some cases.

    !!! note "Dimensions"

        Whether this function computes a vector or matrix norm is determined
        as follows:

        * If `dim` is an `int`, the vector norm will be computed.
        * If `dim` is a `2-tuple`, the matrix norm will be computed.
        * If `dim=None` and ord= None, A will be flattened to 1D and
            the 2-norm of the resulting vector will be computed.
        * If `dim=None` and `ord != None`, A must be 1D or 2D.

    !!! info "See also"

        * `linalg.vector_norm` : computes vector norms
        * `linalg.matrix_norm` : computes matrix norms

        The above functions are often clearer and more flexible than
        using `linalg.norm()`. For example:

        * `linalg.norm(A, ord=1, dim=(0, 1))`  always computes a
            matrix norm, but with
        * `linalg.vector_norm(A, ord=1, dim=(0, 1))` it is possible to
            compute a vector norm over the two dimensions.

    History
    -------
    !!! added "1.7"
        `torch.linalg.norm` was added and `torch.norm` was deprecated
        in torch version `1.7`. `torch.linalg.norm`  has the same signature
        and behavior as `torch.norm`, except that the order is named `ord`
        instead of `p`.
    !!! added "1.1"
        The `dtype` option was added in torch version `1.1`.
    !!! added "1.0"
        The infinite, Frobenius and nuclear norms were added in
        torch version `1.0`. At this point, a vector or matrix norm was
        chosen based on the type of norm (`p`) and `dim` value.
    !!! added "0.2"
        The `keepdim` option was added in torch version `0.2`.
    !!! added "0.1"
        `torch.norm` exists since at least torch version `0.1`.
        At that time, it always computed a vector norm and its signature
        was `torch.norm(x, p: float = 2)` or
        `torch.norm(x, p: float, dim: int, out=None)`.

    Parameters
    ----------
    A : tensor
        Tensor of shape `(*, n)` or `(*, m, n)` where `*` is zero or
        more batch dimensions.
    ord : "{int, float, inf, -inf}"
        The order of norm.
    dim :  int or tuple[int]
        Dimensions over which to compute the vector or matrix norm.
        See above for the behavior when `dim=None`.
    keepdim : bool
        If set to True, the reduced dimensions are retained in the
        result as dimensions with size one.

    Other Parameters
    ----------------
    out : tensor
        The output tensor. Ignored if `None`.
    dtype : torch.dtype
        If specified, the input tensor is cast to `dtype` before
        performing the operation, and the returned tensor's type
        will be `dtype`.

    Returns
    -------
    out : tensor
        A real-valued tensor, even when `A` is complex.
    """
    ...


def vector_norm(
        x, ord=2, dim=None, keepdim=False, *, out=None, dtype=None
):
    """Computes a vector norm.

    `ord` defines the norm that is computed. The following norms are supported:

    |  `ord`  |          vector norm          |
    | ------- | ----------------------------- |
    |  `inf`  | `max(abs(x))`                 |
    | `-inf`  | `min(abs(x))`                 |
    |  `0`    | `sum(x != 0)`                 |
    | other   | `sum(abs(x)**ord)**(1./ord)`  |

    !!! note "Data types"

        Supports input of `float`, `double`, `cfloat` and `cdouble` dtypes.

        If `x` is complex valued, it computes the norm of `x.abs()`.

        `dtype` may be used to perform the computation in a more precise
        dtype. It is semantically equivalent to calling
        `linalg.vector_norm(x.to(dtype))` but it is faster in some cases.

    !!! note "Dimensions"

        This function does not necessarily treat multidimensional `x` as
        a batch of vectors, instead:

        * If `dim=None`, `x` will be flattened before the norm is computed.
        * If `dim` is an `int` or a `tuple`, the norm will be computed over
          these dimensions and the other dimensions will be treated as
          batch dimensions.

        This behavior is for consistency with `linalg.norm()`.

    !!! info "See also"

        * `linalg.norm` : computes matrix and vector norms
        * `linalg.matrix_norm` : computes matrix norms

    History
    -------
    !!! added "1.9"
        `torch.linalg.vector_norm` was added in torch version `1.9`.

    Parameters
    ----------
    x : tensor
        Tensor of shape `(*, n)` where `*` is zero or more batch dimensions.
    ord : "{int, float, inf, -inf}"
        The order of norm.
    dim :  int or tuple[int]
        Dimensions over which to compute the vector norm.
    keepdim : bool
        If set to True, the reduced dimensions are retained in the
        result as dimensions with size one.

    Other Parameters
    ----------------
    out : tensor
        The output tensor. Ignored if `None`.
    dtype : torch.dtype
        If specified, the input tensor is cast to `dtype` before
        performing the operation, and the returned tensor's type
        will be `dtype`.

    Returns
    -------
    out : tensor
        A real-valued tensor, even when `x` is complex.

    """
    ...


def matrix_norm(
        A, ord='fro', dim=(-2, -1), keepdim=False, *, out=None, dtype=None
):
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

    !!! note "Data types"

        Supports input of `float`, `double`, `cfloat` and `cdouble` dtypes.

        If `A` is complex valued, it computes the norm of `A.abs()`.

        `dtype` may be used to perform the computation in a more precise
        dtype. It is semantically equivalent to calling
        `linalg.matrix_norm(A.to(dtype))` but it is faster in some cases.

    !!! note "Dimensions"

        Also supports batches of matrices: the norm will be computed over
        the dimensions specified by the 2-tuple `dim` and the other dimensions
        will be treated as batch dimensions. The output will have the same
        batch dimensions.

    !!! info "See also"

        * `linalg.norm` : computes matrix and vector norms
        * `linalg.vector_norm` : computes vector norms

    History
    -------
    !!! added "1.9"
        `torch.linalg.vector_norm` was added in torch version `1.9`.

    Parameters
    ----------
    A : tensor
        Tensor with two or more dimensions. By default its shape is
        interpreted as `(*, m, n)` where `*` is zero or more batch
        dimensions, but this behavior can be controlled using `dim`.
    ord : {int, float, inf, -inf}
        The order of norm.
    dim :  int or tuple[int]
        Dimensions over which to compute the matrix norm.
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
        A real-valued tensor, even when `A` is complex.

    """
    ...


def diagonal(A, *, offset=0, dim1=-2, dim2=-1):
    """
    Returns a partial view of input with the its diagonal elements with
    respect to dim1 and dim2 appended as a dimension at the end of the shape.

    The argument offset controls which diagonal to consider:

    * If `offset = 0`, it is the main diagonal.
    * If `offset > 0`, it is above the main diagonal.
    * If `offset < 0`, it is below the main diagonal.

    Applying `torch.diag_embed` to the output of this function with the
    same arguments yields a diagonal matrix with the diagonal entries of
    the input.

    History
    -------
    !!! added "1.11"
        `torch.linalg.diagonal` was added in torch version `1.11`.

    Parameters
    ----------
    A : tensor
        The input tensor. Must be at least 2-dimensional.

    Other Parameters
    ----------------
    offset : int
        Which diagonal to consider.
    dim1 : int
        First dimension with respect to which to take diagonal.
    dim2 : int
        Second dimension with respect to which to take diagonal.

    Returns
    -------
    out : tensor
        Diagonal of `A`.

    """
    ...


def det(A, *, out=None):
    """
    Computes the determinant of a square matrix.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if `A` is a batch of
        matrices then the output has the same batch dimensions.

    !!! info "See also"
        `linalg.slogdet` computes the sign and natural logarithm of
        the absolute value of the determinant of square matrices.

    History
    -------
    !!! added "1.7"
        `torch.linalg.det` was added in torch version `1.7`, as an
        alias for `torch.det`.

    Parameters
    ----------
    A : tensor
        Tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.

    Other Parameters
    ----------------
    out : tensor
        Output tensor. Ignored if `None`.

    Returns
    -------
    out : tensor
        Determinant of `A`.

    """
    ...


def slogdet(A, *, out=None):
    """
    Computes the sign and natural logarithm of the absolute value of
    the determinant of a square matrix.

    For complex `A`, it returns the sign and the natural logarithm of the
    modulus of the determinant, that is, a logarithmic polar decomposition
    of the determinant.

    The determinant can be recovered as `sign * exp(logabsdet)`.
    When a matrix has a determinant of zero, it returns `(0, -inf)`.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if `A` is a batch of
        matrices then the output has the same batch dimensions.

    !!! warning "Backward"
        Backward through `slogdet` internally uses SVD results when
        input is not invertible. In this case, double backward through
        `slogdet` will be unstable in when input doesn't have distinct
        singular values. See `svd` for details.

    !!! info "See also"
        `linalg.det` computes the determinant of square matrices.

    History
    -------
    !!! added "1.8"
        `torch.linalg.slogdet` was added in torch version `1.8`, as an
        alias for `torch.slogdet`.

    Parameters
    ----------
    A : tensor
        Tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.

    Other Parameters
    ----------------
    out : tensor
        output tuple of two tensors. Ignored if `None`.

    Returns
    -------
    sign : tensor
        Sign of `det(A)`.
    logabsdet : tensor
        Log of `det(A)`, will always be real-valued, even when `A` is complex.

    """
    ...


def cond(A, ord=2, *, out=None):
    r"""
    Computes the condition number of a matrix with respect to a matrix norm.

    Letting $\mathbb{K}$ be $\mathbb{R}$ or $\mathbb{C}$, the
    **condition number** $\kappa$ of a matrix $A \in \mathbb{K}^{n \times n}$
    is defined as

    $$\kappa(A) = \lVert A \rVert_p \lVert A^{-1} \rVert_p$$

    The condition number of `A` measures the numerical stability of the
    linear system `AX = B` with respect to a matrix norm.

    `ord` defines the matrix norm that is computed. The following norms
    are supported:

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

    For `ord` is one of `{'fro', 'nuc', inf, -inf, 1, -1}`, this
    function uses `linalg.norm` and `linalg.inv`. As such, in this case,
    the matrix (or every matrix in the batch) `A` has to be square and
    invertible.

    For `ord` in `{-2, 2}`, this function can be computed in terms of the
    singular values $\sigma_1 \leq \dots \leq \sigma_n$

    $$\kappa_2(A) = \frac{\sigma_1}{\sigma_n} ~~~
    \kappa_{-2}(A) = \frac{\sigma_n}{\sigma_1}$$

    In these cases, it is computed using `torch.linalg.svdvals`.
    For these norms, the matrix (or every matrix in the batch) `A` may
    have any shape.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    !!! note "Device"
        When inputs are on a CUDA device, this function synchronizes that
        device with the CPU if p is one of `{'fro', 'nuc', inf, -inf, 1, -1}`.

    !!! info "See also"
        * `linalg.solve` for a function that solves linear systems of
          square matrices.
        * `linalg.lstsq` for a function that solves linear systems of
          general matrices.

    History
    -------
    !!! added "1.8"
        `torch.linalg.cond` was added in torch version `1.8`.

    Parameters
    ----------
    A : tensor
        Tensor of shape `(*, m, n)` where `*` is zero or more batch
        dimensions for `ord` in `{2, -2}`, and of shape `(*, n, n)` where
        every matrix is invertible for `ord` in
        `{'fro', 'nuc', inf, -inf, 1, -1}`.
    ord : {int, inf, -inf, 'fro', 'nuc'}
        The type of the matrix norm to use in the computations.

    Other Parameters
    ----------------
    out : tensor
        Output tensor. Ignored if `None`.

    Returns
    -------
    out : tensor
        A real-valued tensor, even when `A` is complex.

    """
    ...


def matrix_rank(A, *, atol=None, rtol=None, hermitian=False, out=None):
    r"""
    Computes the numerical rank of a matrix.

    The matrix rank is computed as the number of singular values
    (or eigenvalues in absolute value when hermitian= True) that are
    greater than $\text{max}(\text{atol}, \sigma_1 * \text{rtol})$
    threshold, where $\sigma_1$ is the largest singular value (or eigenvalue).

    !!! warning
        If `torch < 1.11`, `rtol` is not used.

    !!! warning
        If `hermitian=True`, `A` is assumed to be Hermitian if complex or
        symmetric if real, but this is not checked internally. Instead,
        just the lower triangular part of the matrix is used in the
        computations.

    !!! info
        If `rtol` is not specified and `A` is a matrix of dimensions
        `(m, n)`, the relative tolerance is set to be
        $\text{rtol} = \text{max}(m, n) \varepsilon$, and $\varepsilon$
        is the epsilon value for the dtype of `A`.

        If `rtol` is not specified and `atol` is specified to be larger
        than zero then `rtol` is set to zero.

        If `atol` or `rtol` is a tensor, its shape must be broadcastable
        to that of the singular values of `A` as returned by `linalg.svdvals`.

    !!! info
        The matrix rank is computed using a singular value decomposition
        `linalg.svdvals` if `hermitian=False` (default) and the
        eigenvalue decomposition `linalg.eigvalsh` when `hermitian=True`.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    !!! note "Device"
        When inputs are on a CUDA device, this function synchronizes that
        device with the CPU.

    History
    -------
    !!! added "1.11"
        `tol` was replaced with `atol` and `rtol` in torch version `1.11`.
    !!! added "1.8"
        `torch.linalg.matrix_rank` was added and `torch.matrix_rank` was
        deprecated in torch version `1.8`. At that time, its signature was
        `linalg.matrix_rank(input, tol=None, hermitian=False, *, out=None)`.

        `out` was added to `torch.matrix_rank`. At that time, its signature was
        `matrix_rank(input, tol=None, symmetric=False, *, out=None)`.
    !!! added "1.0"
        `torch.matrix_rank` was added in torch version `1.0`.
        At that time, its signature was
        `matrix_rank(input, tol=None, symmetric=False,)`.

    Parameters
    ----------
    A : tensor
        Tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.

    Other Parameters
    ----------------
    atol : float or Tensor
        The absolute tolerance value. When None it's considered to be zero.
    rtol : float or Tensor
        The relative tolerance value.
    hermitian : bool
        Indicates whether `A` is Hermitian if complex or symmetric if real.
    out : tensor
        Output tensor. Ignored if `None`.

    Returns
    -------
    rank : tensor
        Matrix rank
    """
    ...


def cholesky(A, *, upper=False, out=None):
    r"""
    Computes the Cholesky decomposition of a complex Hermitian or real
    symmetric positive-definite matrix.

    Letting $\mathbb{K}$ be $\mathbb{R}$ or $\mathbb{C}$, the
    **Cholesky decomposition** of a complex Hermitian or real symmetric
    positive-definite matrix $A \in \mathbb{K}^{n \times n}$ is defined as

    $$A = LL^{\text{H}} ~~~~~~ L \in \mathbb{K}^{n \times n}$$

    where $L$ is a lower triangular matrix with real positive diagonal
    (even in the complex case) and $L^{\text{H}}$ is the conjugate transpose
    when $L$ is complex, and the transpose when $L$ is real-valued.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    !!! note "Device"
        When inputs are on a CUDA device, this function synchronizes that
        device with the CPU.

    History
    -------
    !!! added "1.10"
        Option `upper` was added to `torch.linalg.cholesky` in
        torch version `1.10`.
    !!! added "1.8"
        `torch.linalg.cholesky` was added and `torch.cholesky` was
        deprecated in torch version `1.8`. At that time, its signature was
        `linalg.cholesky(input, *, out=None)`.
    !!! added "1.0"
        `torch.cholesky` was added in torch version `1.0` in place of
        `torch.potrf`, with the signature `cholesky(x, upper=True, out=None)`.
    !!! added "0.1"
        `torch.potrf` was added in torch version `0.1`, with the
        signature `potrf(x, upper=True, out=None)`.

    Parameters
    ----------
    A : tensor
        tensor of shape `(*, n, n)` where `*` is zero or more batch
        dimensions consisting of symmetric or Hermitian
        positive-definite matrices.

    Other Parameters
    ----------------
    upper : bool
        Whether to return an upper triangular matrix.
        The tensor returned with `upper=True` is the conjugate transpose
        of the tensor returned with `upper=False`.
    out : tensor
        Output tensor. Ignored if `None`.

    Returns
    -------
    out : tensor
        Cholesky decomposition
    """
    ...


def qr(A, mode='reduced', *, out=None):
    r"""
    Computes the QR decomposition of a matrix.

    Letting $\mathbb{K}$ be $\mathbb{R}$ or $\mathbb{C}$, the
    **full QR decomposition** of a matrix $A \in \mathbb{K}^{m \times n}$
    is defined as

    $$A = QR ~~~~~~
    Q \in \mathbb{K}^{m \times m}, R \in \mathbb{K}^{m \times n}$$

    where $Q4 is orthogonal in the real case and unitary in the complex case,
    and $R$ is upper triangular with real diagonal (even in the complex case).

    When $m > n$ (tall matrix), as $R$ is upper triangular, its last $m - n$
    rows are zero. In this case, we can drop the last $m - n$ columns of $Q$
    to form the **reduced QR decomposition**:

    $$A = QR ~~~~~~
    Q \in \mathbb{K}^{m \times n}, R \in \mathbb{K}^{n \times n}$$

    The reduced QR decomposition agrees with the full QR decomposition
    when $n \geq m$ (wide matrix).

    !!! note "Dimensions"

        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

        The parameter mode chooses between the full and reduced QR
        decomposition. If `A` has shape `(*, m, n)`, denoting `k = min(m, n)`

        * `mode='reduced'`: Returns `(Q, R)` of shapes `(*, m, k)`, `(*, k, n)`
          respectively. It is always differentiable.
        * `mode='complete'`: Returns `(Q, R)` of shapes `(*, m, m)`,
          `(*, m, n)` respectively. It is differentiable for `m <= n`.
        * `mode='r'`: Computes only the reduced `R`. Returns `(Q, R)`
          with `Q` empty and `R` of shape `(*, k, n)`.
          It is never differentiable.

    !!! warning "Differences with `numpy.linalg.qr`"

        * `mode='raw'` is not implemented.

        Unlike `numpy.linalg.qr`, this function always returns a tuple
        of two tensors. When `mode='r'`, the `Q` tensor is an empty tensor.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! warning
        The elements in the diagonal of R are not necessarily positive.
        As such, the returned QR decomposition is only unique up to the
        sign of the diagonal of R. Therefore, different platforms, like
        NumPy, or inputs on different devices, may produce different
        valid decompositions.

    !!! warning
        The QR decomposition is only well-defined if the first `k = min(m, n)`
        columns of every matrix in A are linearly independent. If this
        condition is not met, no error will be thrown, but the QR produced
        may be incorrect and its autodiff may fail or produce incorrect
        results.

    History
    -------
    !!! added "1.8"
        `torch.linalg.qr` was added and `torch.qr` was deprecated in
        torch version `1.8`.
    !!! added "1.2"
        Option `some` was added to `torch.qr` in torch version `1.2`.
    !!! added "0.1"
        `torch.qr` is available since torch version `0.1`.

    Parameters
    ----------
    A : tensor
        Tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    mode : {'reduced', 'complete', 'r'}
        Controls the shape of the returned tensors.

    Other Parameters
    ----------------
    out : tuple[tensor]
        Output tuple of two tensors. Ignored if `None`.

    Returns
    -------
    Q : tensor
    R : tensor

    """
    ...


def lu(A, *, pivot=True, out=None):
    r"""
    Computes the LU decomposition with partial pivoting of a matrix.

    Letting $\mathbb{K}$ be $\mathbb{R}$ or $\mathbb{C}$, the
    **LU decomposition with partial pivoting** of a matrix
    $A \in \mathbb{K}^{m \times n}$ is defined as

    $$A = PLU ~~~~~~
    P \in \mathbb{K}^{m \times m},
    L \in \mathbb{K}^{m \times k},
    U \in \mathbb{K}^{k \times n}$$$

    where `k = min(m,n)`, $P$ is a permutation matrix, $L$ is lower
    triangular with ones on the diagonal and $U$ is upper triangular.

    If `pivot=False` and `A` is on GPU, then the LU decomposition
    without pivoting is computed

    $$A = LU ~~~~~~
    L \in \mathbb{K}^{m \times k},
    U \in \mathbb{K}^{k \times n}$$$

    When `pivot=False`, the returned matrix `P` will be empty.
    The LU decomposition without pivoting may not exist if any of the
    principal minors of `A` is singular. In this case, the output matrix
    may contain `inf` or `NaN`.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    !!! warning
        The LU decomposition is almost never unique, as often there are
        different permutation matrices that can yield different LU
        decompositions. As such, different platforms, like SciPy, or
        inputs on different devices, may produce different valid
        decompositions.

    !!! warning
        Gradient computations are only supported if the input matrix is
        full-rank. If this condition is not met, no error will be thrown,
        but the gradient may not be finite. This is because the LU
        decomposition with pivoting is not differentiable at these points.

    !!! info "See also"
        `linalg.solve` solves a system of linear equations using the
        LU decomposition with partial pivoting.

    History
    -------
    !!! added "1.12"
        `torch.linalg.lu` was added in torch version `1.12`.
    !!! added "1.1"
        `torch.lu` was added and `torch.btrifactor` was deprecated in
        torch version `1.1`.
    !!! added "0.1"
        `torch.btrifactor` was available in torch version `0.1`.

    Parameters
    ----------
    A : tensor
        Tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.

    Other Parameters
    ----------------
    pivot : bool
        Controls whether to compute the LU decomposition with partial
        pivoting or no pivoting.
    out : tuple[tensor]
        Output tuple of three tensors. Ignored if `None`.

    Returns
    -------
    P : tensor
    L : tensor
    U : tensor
    """
    ...


def lu_factor(A, *, pivot=True, out=None):
    r"""
    Computes a compact representation of the LU factorization with
    partial pivoting of a matrix.

    This function computes a compact representation of the decomposition
    given by `linalg.lu`. If the matrix is square, this representation
    may be used in `linalg.lu_solve` to solve system of linear equations
    that share the matrix `A`.

    The returned decomposition is represented as a named tuple `(LU, pivots)`.
    The `LU` matrix has the same shape as the input matrix `A`.
    Its upper and lower triangular parts encode the non-constant elements
    of `L` and `U` of the LU decomposition of `A`.

    The returned permutation matrix is represented by a 1-indexed vector.
    `pivots[i] == j` represents that in the `i`-th step of the algorithm,
    the `i`-th row was permuted with the `j-1`-th row.

    On CUDA, one may use `pivot=False`. In this case, this function
    returns the LU decomposition without pivoting if it exists.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    !!! note "Device"
        When inputs are on a CUDA device, this function synchronizes
        that device with the CPU.

    !!! warning
        The LU decomposition is almost never unique, as often there are
        different permutation matrices that can yield different LU
        decompositions. As such, different platforms, like SciPy, or
        inputs on different devices, may produce different valid
        decompositions.

    !!! warning
        Gradient computations are only supported if the input matrix is
        full-rank. If this condition is not met, no error will be thrown,
        but the gradient may not be finite. This is because the LU
        decomposition with pivoting is not differentiable at these points.

    !!! info "See also"
        * `linalg.lu_solve` solves a system of linear equations given the
          output of this function provided the input matrix was square
          and invertible.
        * `lu_unpack` unpacks the tensors returned by `lu_factor` into
          the three matrices P, L, U that form the decomposition.
        * `linalg.lu` computes the LU decomposition with partial pivoting
          of a possibly non-square matrix. It is a composition of
          `lu_factor` and `lu_unpack`.
        * `linalg.solve` solves a system of linear equations.
          It is a composition of `lu_factor` and `lu_solve`.

    History
    -------
    !!! added "1.11"
        `torch.linalg.lu_factor` was added in torch version `1.11`.
    !!! added "1.1"
        `torch.lu` was added and `torch.btrifactor` was deprecated in
        torch version `1.1`.
    !!! added "0.1"
        `torch.btrifactor` was available in torch version `0.1`.

    Parameters
    ----------
    A : tensor
        tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.

    Other Parameters
    ----------------
    pivot : bool
        Controls whether to compute the LU decomposition with partial
        pivoting or no pivoting.
    out : tuple[tensor]
        Output tuple of two tensors. Ignored if `None`.

    Returns
    -------
    LU : tensor
    pivots : tensor
    """
    ...


def eig(A, *, out=None):
    r"""
    Computes the eigenvalue decomposition of a square matrix if it exists.

    Letting $\mathbb{K}$ be $\mathbb{R}$ or $\mathbb{C}$, the
    **eigenvalue decomposition** of a matrix
    $A \in \mathbb{K}^{n \times n}$ (if it exists) is defined as

    $$A = V \text{diag}(\Lambda) V^{-1} ~~~~~~
    V \in \mathbb{C}^{n \times n}, \Lambda \in \mathbb{C}^{n}$$

    This decomposition exists if and only if $A$ is diagonalizable.
    This is the case when all its eigenvalues are different.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    !!! note "Device"
        When inputs are on a CUDA device, this function synchronizes
        that device with the CPU.

    !!! note
        The eigenvalues and eigenvectors of a real matrix may be complex.

    !!! warning
        This function assumes that `A` is diagonalizable (for example,
        when all the eigenvalues are different). If it is not diagonalizable,
        the returned eigenvalues will be correct but
        $A \neq V \text{diag}(\Lambda) V^{-1}$.

    !!! warning
        The returned eigenvectors are normalized to have norm 1. Even then,
        the eigenvectors of a matrix are not unique, nor are they continuous
        with respect to A. Due to this lack of uniqueness, different hardware
        and software may compute different eigenvectors.

        This non-uniqueness is caused by the fact that multiplying an
        eigenvector by $e^{i\phi}, \phi \in \mathbb{R}$ produces another
        set of valid eigenvectors of the matrix. For this reason, the loss
        function shall not depend on the phase of the eigenvectors, as
        this quantity is not well-defined. This is checked when computing
        the gradients of this function. As such, when inputs are on a CUDA
        device, this function synchronizes that device with the CPU when
        computing the gradients. This is checked when computing the gradients
        of this function. As such, when inputs are on a CUDA device, the
        computation of the gradients of this function synchronizes that
        device with the CPU.

    !!! warning
        Gradients computed using the eigenvectors tensor will only be finite
        when `A` has distinct eigenvalues. Furthermore, if the distance
        between any two eigenvalues is close to zero, the gradient will
        be numerically unstable, as it depends on the eigenvalues
        $\lambda_i$ through the computation of
        $\frac{1}{\min_{i\neq j}\lambda_i - \lambda_j}$.

    !!! info "See also"
        * `linalg.eigvals` computes only the eigenvalues. Unlike
          `linalg.eig`, the gradients of `eigvals` are always numerically
          stable.
        * `linalg.eigh` for a (faster) function that computes the eigenvalue
          decomposition for Hermitian and symmetric matrices.
        * `linalg.svd` for a function that computes another type of spectral
          decomposition that works on matrices of any shape.
        * `linalg.qr` for another (much faster) decomposition that works
          on matrices of any shape.

    History
    -------
    !!! added "1.9"
        `torch.linalg.eig` was added and `torch.eig` was deprecated in
        torch version `1.9`.
    !!! added "0.1"
        `torch.eig` is available since torch version `0.1`.

    Parameters
    ----------
    A : tensor
        Tensor of shape `(*, n, n)` where `*` is zero or more batch
        dimensions consisting of diagonalizable matrices.

    Other Parameters
    ----------------
    out : tuple[tensor]
        Output tuple of two tensors. Ignored if `None`.

    Returns
    -------
    eigenvalues : tensor
    eigenvectors : tensor
    """
    ...


def eigvals(A, *, out=None):
    r"""
    Computes the eigenvalues of a square matrix.

    Letting $\mathbb{K}$ be $\mathbb{R}$ or $\mathbb{C}$, the
    **eigenvalues** of a square matrix
    $A \in \mathbb{K}^{n \times n}$ are defined as the roots
    (counted with multiplicity) of the polynomial `p` of degree `n` given by

    $$p(\lambda) = \text{det}(A - \lambda I_n) ~~~~~~ \lambda \in \mathbb{C}$$

    where $I_n$ is the `n`-dimensional identity matrix.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    !!! note "Device"
        When inputs are on a CUDA device, this function synchronizes
        that device with the CPU.

    !!! note
        The eigenvalues of a real matrix may be complex, as the roots of
        a real polynomial may be complex.

        The eigenvalues of a matrix are always well-defined, even when
        the matrix is not diagonalizable.

    !!! info "See also"
        `linalg.eig` computes the full eigenvalue decomposition.


    History
    -------
    !!! added "1.9"
        `torch.linalg.eigvals` was added in torch version `1.9`.

    Parameters
    ----------
    A : tensor
        Tensor of shape `(*, n, n)` where `*` is zero or more batch
        dimensions.

    Other Parameters
    ----------------
    out : tensor
        Output tensor. Ignored if `None`.

    Returns
    -------
    eigenvalues : tensor
        A complex-valued tensor containing the eigenvalues even when
        `A` is real.
    """
    ...


def eigh(A, UPLO='L', *, out=None):
    r"""
    Computes the eigenvalue decomposition of a complex Hermitian or real
    symmetric matrix.

    Letting $\mathbb{K}$ be $\mathbb{R}$ or $\mathbb{C}$, the
    **eigenvalue decomposition** of a complex Hermitian or real symmetric
    matrix $A \in \mathbb{K}^{n \times n}$ is defined as

    $$A = Q \text{diag}(\Lambda) Q^{\text{H}} ~~~~~~
    Q \in \mathbb{K}^{n \times n}, \Lambda \in \mathbb{R}^{n}$$

    where $Q^{\text{H}}$ is the conjugate transpose when $Q$ is complex, and
    the transpose when $Q$ is real-valued. $Q$ is orthogonal in the real
    case and unitary in the complex case.

    !!! warning
        `A` is assumed to be Hermitian (resp. symmetric), but this is not
        checked internally, instead:

        * If `UPLO='L'`, only the lower triangular part of the matrix is
          used in the computation.
        * If `UPLO='U'`, only the upper triangular part of the matrix is used.

    !!! note
        The eigenvalues of real symmetric or complex Hermitian matrices
        are always real.

        The eigenvalues are returned in ascending order.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    !!! note "Device"
        When inputs are on a CUDA device, this function synchronizes
        that device with the CPU.

    !!! warning
        The eigenvectors of a symmetric matrix are not unique, nor are
        they continuous with respect to `A`. Due to this lack of
        uniqueness, different hardware and software may compute
        different eigenvectors.

        This non-uniqueness is caused by the fact that multiplying an
        eigenvector by $e^{i\phi}, \phi \in \mathbb{R}$ produces another
        set of valid eigenvectors of the matrix. For this reason, the loss
        function shall not depend on the phase of the eigenvectors, as
        this quantity is not well-defined. This is checked when computing
        the gradients of this function. As such, when inputs are on a CUDA
        device, this function synchronizes that device with the CPU when
        computing the gradients. This is checked when computing the gradients
        of this function. As such, when inputs are on a CUDA device, the
        computation of the gradients of this function synchronizes that
        device with the CPU.

    !!! warning
        Gradients computed using the eigenvectors tensor will only be finite
        when `A` has distinct eigenvalues. Furthermore, if the distance
        between any two eigenvalues is close to zero, the gradient will
        be numerically unstable, as it depends on the eigenvalues
        $\lambda_i$ through the computation of
        $\frac{1}{\min_{i\neq j}\lambda_i - \lambda_j}$.

    !!! info "See also"
        * `linalg.eigvalsh` computes only the eigenvalues of a Hermitian
          matrix. Unlike torch.linalg.eigh(), the gradients of `eigvalsh`
          are always numerically stable.
        * `linalg.cholesky` for a different decomposition of a Hermitian
          matrix. The Cholesky decomposition gives less information about
          the matrix but is much faster to compute than the eigenvalue
          decomposition.
        * `linalg.eig` for a (slower) function that computes the eigenvalue
          decomposition of a not necessarily Hermitian square matrix.
        * `linalg.svd` for a (slower) function that computes the more general
          SVD decomposition of matrices of any shape.
        * `linalg.qr` for another (much faster) decomposition that works
          on general matrices.

    History
    -------
    !!! added "2.0"
        `torch.symeig` was removed in torch version `2.0`.
    !!! added "1.9"
        `torch.symeig` was deprecated in torch version `1.9`.
    !!! added "1.8"
        `torch.linalg.eigh` was added in torch version `1.8`.
    !!! added "0.1"
        `torch.symeig` was added in torch version `0.1`.

    Parameters
    ----------
    A : tensor
        Tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
        consisting of symmetric or Hermitian matrices.
    UPLO : {'L', 'U'}
        Controls whether to use the upper or lower triangular part of `A`
        in the computations.

    Other Parameters
    ----------------
    out : tuple[tensor]
        Output tuple of two tensors. Ignored if `None`.

    Returns
    -------
    eigenvalues : tensor
        It will always be real-valued, even when A is complex.
        It will also be ordered in ascending order.
    eigenvectors : tensor
        It will have the same dtype as A and will contain the eigenvectors
        as its columns.
    """
    ...


def eigvalsh(A, UPLO='L', *, out=None):
    r"""
    Computes the eigenvalues of a complex Hermitian or real
    symmetric matrix.

    Letting $\mathbb{K}$ be $\mathbb{R}$ or $\mathbb{C}$, the
    **eigenvalues** of a complex Hermitian or real symmetric
    matrix $A \in \mathbb{K}^{n \times n}$ are defined as the roots
    (counted with multiplicity) of the polynomial `p` of degree `n` given by

    $$p(\lambda) = \text{det}(A - \lambda I_n) ~~~~~~ \lambda \in \mathbb{C}$$

    where $I_n$ is the `n`-dimensional identity matrix.

    !!! warning
        `A` is assumed to be Hermitian (resp. symmetric), but this is not
        checked internally, instead:

        * If `UPLO='L'`, only the lower triangular part of the matrix is
          used in the computation.
        * If `UPLO='U'`, only the upper triangular part of the matrix is used.

    !!! note
        The eigenvalues of real symmetric or complex Hermitian matrices
        are always real.

        The eigenvalues are returned in ascending order.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    !!! note "Device"
        When inputs are on a CUDA device, this function synchronizes
        that device with the CPU.

    !!! info "See also"
        `linalg.eigh`  computes the full eigenvalue decomposition.

    History
    -------
    !!! added "2.0"
        `torch.symeig` was removed in torch version `2.0`.
    !!! added "1.9"
        `torch.symeig` was deprecated in torch version `1.9`.
    !!! added "1.8"
        `torch.linalg.eigvalsh` was added in torch version `1.8`.
    !!! added "0.1"
        `torch.symeig` was added in torch version `0.1`.

    Parameters
    ----------
    A : tensor
        Tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
        consisting of symmetric or Hermitian matrices.
    UPLO : {'L', 'U'}
        Controls whether to use the upper or lower triangular part of `A`
        in the computations.

    Other Parameters
    ----------------
    out : tensor
        Output tensor. Ignored if `None`.

    Returns
    -------
    eigenvalues : tensor
        A real-valued tensor containing the eigenvalues even when `A` is
        complex. The eigenvalues are returned in ascending order.
    """
    ...


def svd(A, full_matrices=True, *, out=None):
    r"""
    Computes the singular value decomposition (SVD) of a matrix.

    Letting $\mathbb{K}$ be $\mathbb{R}$ or $\mathbb{C}$, the
    **full SVD** of a matrix $A \in \mathbb{K}^{m \times n}$, if
    `k = min(m, n)`, is defined as

    $$A = U \text{diag}(S) V^{\text{H}} ~~~~~~
    U \in \mathbb{K}^{m \times m},
    S \in \mathbb{R}^k,
    V \in \mathbb{K}^{n \times n}$$

    where $\text{diag}(S) \in \mathbb{K}^{m \times n}$, $V^{\text{H}}$ is
    the conjugate transpose when $V$ is complex, and the transpose when $V$
    is real-valued. The matrices $U$, $V$ (and thus $V^{\text{H}}$) are
    orthogonal in the real case, and unitary in the complex case.

    When `m > n` (resp. `m < n`) we can drop the last `m-n` (resp. `n-m`)
    columns of `U` (resp `V`) to form the **reduced SVD**:

    $$A = U \text{diag}(S) V^{\text{H}} ~~~~~~
    U \in \mathbb{K}^{m \times k},
    S \in \mathbb{R}^k,
    V \in \mathbb{K}^{n \times k}$$

    where $\text{diag}(S) \in \mathbb{K}^{k \times k}. In this case,
    $U$ and $V$ also have orthonormal columns.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    !!! warning "Differences with `numpy.linalg.svd`"
        Unlike `numpy.linalg.svd`, this function always returns a tuple
        of three tensors and it doesn't support `compute_uv` argument.
        Please use `linalg.svdvals`, which computes only the singular
        values, instead of `compute_uv=False`.

    !!! note
        When `full_matrices=True`, the gradients with respect to
        `U[..., :, min(m, n):]` and `Vh[..., min(m, n):, :]` will be ignored,
        as those vectors can be arbitrary bases of the corresponding subspaces.

    !!! warning
        The returned tensors `U` and `V` are not unique, nor are they
        continuous with respect to `A`. Due to this lack of uniqueness,
        different hardware and software may compute different singular vectors.

        This non-uniqueness is caused by the fact that multiplying any pair
        of singular vectors $u_k, v_k$ by -1 in the real case or by
        $e^{i\phi}, \phi \in \mathbb{R}$ in the complex case produces
        another two valid singular vectors of the matrix. For this reason,
        the loss function shall not depend on this $e^{i\phi}$ quantity,
        as it is not well-defined. This is checked for complex inputs when
        computing the gradients of this function. As such, when inputs are
        complex and are on a CUDA device, the computation of the gradients
        of this function synchronizes that device with the CPU.

    !!! warning
        Gradients computed using `U` or `Vh` will only be finite when `A`
        does not have repeated singular values. If `A` is rectangular,
        additionally, zero must also not be one of its singular values.
        Furthermore, if the distance between any two singular values is
        close to zero, the gradient will be numerically unstable, as it
        depends on the singular values $\sigma_i$ through the computation of
        $\frac{1}{\min_{i\neq j}\sigma^2_i - \sigma^2_j}$.
        In the rectangular case, the gradient will also be numerically
        unstable when `A` has small singular values, as it also depends
        on the computation of $\frac{1}{\sigma_i}$.

    !!! info "See also"
        * `linalg.svdvals` computes only the singular values.
          Unlike `linalg.svd`, the gradients of `svdvals` are always
          numerically stable.
        * `linalg.eig` for a function that computes another type of
          spectral decomposition of a matrix. The eigendecomposition works
          just on square matrices.
        * `linalg.eigh` for a (faster) function that computes the eigenvalue
          decomposition for Hermitian and symmetric matrices.
        * `linalg.qr` for another (much faster) decomposition that works
          on general matrices.

    History
    -------
    !!! added "1.8"
        `torch.linalg.svd` was added and `torch.svd` was deprecated
        in torch version `1.8`.
    !!! added "1.0"
        Option `compute_uv` was added to `torch.svd` in torch version `1.0`.
    !!! added "0.1"
        `torch.svd` was added in torch version `0.1`. At that time, its
        signature was `torch.svd(input, some=True, out=None)`.

    Parameters
    ----------
    A : tensor
        Tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    full_matrices : bool
        Controls whether to compute the full or reduced SVD, and consequently,
        the shape of the returned tensors `U` and `Vh`.

    Other Parameters
    ----------------
    out : tuple[tensor]
        Output tuple of three tensors.  Ignored if `None`.

    Returns
    -------
    U : tensor
        `U` will have the same dtype as `A`.
        The left singular vectors will be given by the columns of `U`.
    S : tensor
        `S` will always be real-valued, even when `A` is complex.
        It will also be ordered in descending order.
    Vh : tensor
        `Vh` will have the same dtype as `A`.
        The right singular vectors will be given by the rows of `Vh`.

    """
    ...


def svdvals(A, *, out=None):
    r"""
    Computes the singular values of a matrix.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    !!! note "Device"
        When inputs are on a CUDA device, this function synchronizes
        that device with the CPU.

    !!! note
        This function is equivalent to NumPy's
        `linalg.svd(A, compute_uv=False)`.

    !!! info "See also"
        `linalg.svd` computes the full singular value decomposition.

    History
    -------
    !!! added "1.9"
        `torch.linalg.svdvals` was added in torch version `1.9`.

    Parameters
    ----------
    A : tensor
        Tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.

    Other Parameters
    ----------------
    out : tensor
        Output tensor.  Ignored if `None`.

    Returns
    -------
    singularvalues : tensor
        `S` will always be real-valued, even when `A` is complex.
        It will also be ordered in descending order.
    """
    ...


def solve(A, B, *, left=True, out=None):
    r"""
    Computes the solution of a square system of linear equations
    with a unique solution.

    Letting $\mathbb{K}$ be $\mathbb{R}$ or $\mathbb{C}$, this function
    computes the solution $X \in \mathbb{K}^{n \times k}$ of the
    **linear system** associated to $A \in \mathbb{K}^{n \times n}$,
    $B \in \mathbb{K}^{n \times k}$, which is defined as

    $$AX = B$$

    If `left=False`, this function returns the matrix
    $X \in \mathbb{K}^{n \times k}$ that solves the system

    $$XA = B, ~~~
    A \in \mathbb{K}^{k \times k}, B \in \mathbb{K}^{n \times k}.$$

    This system of linear equations has one solution if and only if
    $A$ is invertible. This function assumes that $A$ is invertible.

    !!! note
        This function computes `X = A.inverse() @ `B in a faster and more
        numerically stable way than performing the computations separately.

    !!! note
        It is possible to compute the solution of the system  $XA = B$
        by passing the inputs A and B transposed and transposing the
        output returned by this function.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

        Letting `*` be zero or more batch dimensions,

        * If A has shape `(*, n, n)` and B has shape `(*, n)`
          (a batch of vectors) or shape `(*, n, k)`
          (a batch of matrices or “multiple right-hand sides”),
          this function returns X of shape `(*, n)` or `(*, n, k)`
          respectively.
        * Otherwise, if A has shape `(*, n, n)` and B has shape `(n,)`
          or `(n, k)`, B is broadcasted to have shape `(*, n)` or `(*, n, k)`
          respectively. This function then returns the solution of the
          resulting batch of systems of linear equations.

    !!! note "Device"
        When inputs are on a CUDA device, this function synchronizes
        that device with the CPU.

    !!! info "See also"
        `linalg.solve_triangular` computes the solution of a
        triangular system of linear equations with a unique solution.

    History
    -------
    !!! added "2.0"
        `torch.solve` was removed in torch version `2.0`.
    !!! added "1.13"
        The `left` keyword was added in torch version `1.13`.
    !!! added "1.8"
        `torch.linalg.solve(A, B) -> X` was introduced in torch version `1.8`.
        It supersedes the original `torch.solve(B, A) -> (X, LU)`.
    !!! added "1.1"
        `torch.lu` (which computes the LU factorization)  and `torch.lu_solve`
        (which solves the resulting batched triangular system) were added
        in torch version `1.1`, and `torch.btrifactor`/`torch.btrisolve`
        were deprecated. `torch.solve` (which combines both steps) was
        also added.
    !!! added "0.1"
        `torch.btrifactor` (which computes the LU factorization) and
        `torch.btrisolve` (which solves the resulting batched triangular
        system) were available in torch version `0.1`

    Parameters
    ----------
    A : tensor
        Tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
    B : tensor
        Right-hand side tensor of shape `(*, n)` or `(*, n, k)` or
        `(n,)` or `(n, k)` according to the rules described above.

    Other Parameters
    ----------------
    left : bool
        Whether to solve the system $AX=B$ or $XA=B$.
    out
        Output tensor. Ignored if None.

    Returns
    -------
    X : tensor
        Tensor of shape `(*, n, k)`

    Raises
    -------
    RuntimeError
        if the A matrix is not invertible or any matrix in a
        batched A is not invertible.
    """
    ...


def solve_triangular(A, B, *,
                     upper, left=True, unitriangular=False, out=None):
    r"""Computes the solution of a triangular system of linear
    equations with a unique solution.

    Letting $\mathbb{K}$ be $\mathbb{R}$ or $\mathbb{C}$, this function
    computes the solution $X \in \mathbb{K}^{n \times k}$ of the
    **linear system** associated to $A \in \mathbb{K}^{n \times n}$,
    $B \in \mathbb{K}^{n \times k}$, which is defined as

    $$AX = B$$

    If `left=False`, this function returns the matrix
    $X \in \mathbb{K}^{n \times k}$ that solves the system

    $$XA = B, ~~~
    A \in \mathbb{K}^{k \times k}, B \in \mathbb{K}^{n \times k}.$$

    If `upper=True` (resp. `False`) just the upper (resp. lower) triangular
    half of A will be accessed. The elements below the main diagonal will be
    considered to be zero and will not be accessed.

    If `unitriangular=True`, the diagonal of A is assumed to be ones and
    will not be accessed.

    The result may contain `NaN`s if the diagonal of A contains zeros or
    elements that are very close to zero and `unitriangular=False` (default)
    or if the input matrix has very small eigenvalues.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    !!! see also
        `linalg.solve()` computes the solution of a general square system
        of linear equations with a unique solution.

    History
    -------
    !!! added "1.11"
        `torch.linalg.solve_triangular(A, B) -> X` was introduced in
        torch version `1.11`. It supersedes the original
        `torch.triangular_solve(B, A) -> (X, A)`.
    !!! added "1.1"
        `torch.triangular_solve` was introduced in torch version `1.11`,
        and `torch.trtrs` was deprecated.
    !!! added "0.1"
        `torch.trtrs` was introduced in torch version `0.1`, but only
        documented in torch version `0.4`.

    Parameters
    ----------
    A : tensor
        tensor of shape `(*, n, n)` (or (`*, k, k)` if `left=True`)
        where `*` is zero or more batch dimensions.
    B : tensor
        right-hand side tensor of shape `(*, n, k)`.
    upper : bool
        whether A is an upper or lower triangular matrix.
    left : bool
        whether to solve the system $AX=B$ or $XA=B$.
    unitriangular : bool
        if `True`, the diagonal elements of A are assumed to be all equal to 1.
    out
        output tensor. Ignored if None.

    Returns
    -------
    X : tensor
        tensor of shape `(*, n, k)`

    """
    ...


def lu_solve(LU, pivots, B, *, left=True, adjoint=False, out=None):
    r"""
    Computes the solution of a square system of linear equations with a
    unique solution given an LU decomposition.

    Letting $\mathbb{K}$ be $\mathbb{R}$ or $\mathbb{C}$, this function
    computes the solution $X \in \mathbb{K}^{n \times k}$ of the
    **linear system** associated to $A \in \mathbb{K}^{n \times n}$,
    $B \in \mathbb{K}^{n \times k}$, which is defined as

    $$AX = B$$

    where $A$ is given factorized as returned by
    [`lu_factor`](lu_factor).

    If `left=False`, this function returns the matrix
    $X \in \mathbb{K}^{n \times k}$ that solves the system

    $$XA = B, ~~~
    A \in \mathbb{K}^{k \times k}, B \in \mathbb{K}^{n \times k}.$$

    If `adjoint=True` (and `left=True`) given an LU factorization of $A$
    this function returns the $X \in \mathbb{K}^{n \time k}$ that solves
    the system

    $$A^{\text{H}}X = B ~~~~~~
    A \in \mathbb{K}^{k \times k}, B \in \mathbb{K}^{n \times k}$$

    where $A^{\text{H}}$ is the conjugate transpose when $A$ is complex,
    and the transpose when $A$ is real-valued. The `left=False` case is
    analogous.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    History
    -------
    !!! added "1.13"
        `torch.linalg.lu_solve` was introduced and `torch.lu_solve`
        was deprecated in torch version `1.13`.
    !!! added "1.1"
        `torch.lu` (which computes the LU factorization)  and `torch.lu_solve`
        (which solves the resulting batched triangular system) were added
        in torch version `1.1`, and `torch.btrifactor`/`torch.btrisolve`
        were deprecated.
    !!! added "0.1"
        `torch.btrifactor` (which computes the LU factorization) and
        `torch.btrisolve` (which solves the resulting batched triangular
        system) were available in torch version `0.1`

    Parameters
    ----------
    LU : tensor
        Tensor of shape `(*, n, n)` (or `(*, k, k)` if `left=True`)
        where `*` is zero or more batch dimensions as returned by
        [`lu_factor`](lu_factor).
    pivots : tensor
        Tensor of shape `(*, n)` (or `(*, k)` if `left=True`) where
        `*` is zero or more batch dimensions as returned by
        [`lu_factor`](lu_factor).
    B : tensor
        Right-hand side tensor of shape `(*, n, k)`.

    Other Parameters
    ----------------
    left : bool
        Whether to solve the system $AX=B$ or $XA=B$.
    adjoint : bool
        Whether to solve the system $AX=B$ or $A^{\text{H}}X=B$.
    out : tensor
        Output tensor. Ignored if `None`.

    Returns
    -------
    X : tensor
        Solution of shape `(*, n, k)`.
    """
    ...


def lstsq(A, B, rcond=None):
    r"""
    Computes a solution to the least squares problem of a system of
    linear equations.

    Letting $\mathbb{K}$ be $\mathbb{R}$ or $\mathbb{C}$, the
    **least squares problem** for a linear system $AX=B$ with
    $A \in \mathbb{K}^{m \times n}, B \in \mathbb{K}^{m \times k}$
    is defined as

    $$\min_{X\in\mathbb{K}^{n \times k}} \lVert AX - B \rVert_F$$

    where $\lVert \cdot \rVert_F$ denotes the Frobenius norm.

    !!! warning "Drivers"
        For backward compatibility, and contrary to `torch.linalg.lstsq`,
        we do not expose the `driver` option, and therefore let pytorch
        use its defaults (`'gels'` for CUDA tensors, `'gelsy'` for CPU
        tensors). The full list of available drivers is:

        * `'gelsy'`: QR with pivoting. Should be used for well-conditioned
          (its condition number is not too large, or you do not mind some
          precision loss) general matrices.
        * `'gels'`: QR. Should be used for well-conditioned full-rank
          matrices.
        * `'gelsd'`: tridiagonal reduction and SVD. Should be used for
          ill-conditioned problems.
        * `'gelss'`: full SVD. Should be used for large ill-conditioned
          problems, in cases where `'gelsd'` runs into memory issues.

        For CUDA input, the only valid driver is `'gels'`, which assumes
        that A is full-rank.

    !!! note "`rcond`"
        `rcond` is used to determine the effective rank of the matrices in `A`
        when driver is one of (`'gelsy'`, `'gelsd'`, `'gelss`'). In this case,
        if $\sigma_i$ are the singular values of `A` in decreasing order,
        $\sigma_i$ will be rounded down to zero if
        $\sigma_i \leq \text{rcond} \cdot \sigma_1$. If `rcond=None`, `rcond`
        is set to the machine precision of the dtype of `A` times `max(m, n)`.

    !!! note
        This function computes `X = A.pinverse() @ B` in a faster and
        more numerically stable way than performing the computations
        separately.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    !!! warning
        The default value of rcond may change in a future PyTorch release.
        It is therefore recommended to use a fixed value to avoid potential
        breaking changes.

    History
    -------
    !!! added "2.0"
        `torch.lstsq` was removed in torch version `2.0`.
    !!! added "1.9"
        `torch.linalg.lstsq` was introduced and `torch.lstsq`
        was deprecated in torch version `1.9`.
    !!! added "1.2"
        `torch.lstsq` was introduced and `torch.gels`
        was deprecated in torch version `1.2`.
    !!! added "0.1"
        `torch.gels` was introduced in torch version `0.1`.

    Parameters
    ----------
    A : tensor
        LHS tensor of shape `(*, m, n)` where `*` is zero or more
        batch dimensions.
    B : tensor
        RHS tensor of shape `(*, m, k)` where `*` is zero or more
        batch dimensions.
    rcond : float
        Used to determine the effective rank of `A`.
        If `rcond=None`, `rcond` is set to the machine precision of
        the dtype of `A` times `max(m, n)`.

    Returns
    -------
    solution : tensor
        The least squares solution. It has shape `(*, n, k)`.
    residuals : tensor
        The squared residuals of the solutions, that is,
        $$\lVert AX - B \rVert^2_F$$. It has shape equal to the batch
        dimensions of `A`. It is computed when `m > n` and every matrix
        in `A` is full-rank, otherwise, it is an empty tensor.
        If` A` is a batch of matrices and any matrix in the batch is not
        full rank, then an empty tensor is returned. This behavior may
        change in a future PyTorch release.
    rank : tensor
        Tensor of ranks of the matrices in A. It has shape equal to the
        batch dimensions of A. It is computed when driver is one of
        (`'gelsy'`, `'gelsd'`, gelss`'), otherwise it is an empty tensor.
    singular_values : tensor
        Tensor of singular values of the matrices in `A`.
        It has shape `(*, min(m, n))`. It is computed when driver is
        one of (`'gelsd'`, `'gelss'`), otherwise it is an empty tensor.
    """
    ...


def inv(A, *, out=None):
    r"""
    Computes the inverse of a square matrix if it exists.
    Throws a RuntimeError if the matrix is not invertible.

    Letting $\mathbb{K}$ be $\mathbb{R}$ or $\mathbb{C}$, for a matrix
    $A \in \mathbb{K}^{n \times n}$, its **inverse matrix**
    $A^{-1} \in \mathbb{K}^{n \times n}$ (if it exists) is defined as

    $$A^{-1}A = AA^{-1} = I_n$$

    where $I_n$ is the $n$-dimensional identity matrix.

    The inverse matrix exists if and only if $A$ is invertible.
    In this case, the inverse is unique.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    !!! note "Device"
        When inputs are on a CUDA device, this function synchronizes
        that device with the CPU.

    !!! note
        Consider using `linalg.solve` if possible for multiplying a
        matrix on the left by the inverse, as:
        ```python
        linalg.solve(A, B) == linalg.inv(A) @ B
        ```
        It is always preferred to use `solve` when possible, as it is
        faster and more numerically stable than computing the inverse
        explicitly.

    !!! info "See also"
        * `linalg.pinv` computes the pseudoinverse (Moore-Penrose inverse)
          of matrices of any shape.
        * `linalg.solve` computes `A.inv() @ B` with a numerically
          stable algorithm.

    History
    -------
    !!! added "1.8"
        `torch.linalg.inv` was introduced in torch version `1.8`.
    !!! added "0.1"
        `torch.inverse` was introduced in torch version `0.1`.

    Parameters
    ----------
    A : tensor
        Tensor of shape `(*, n, n)` where `*` is zero or more batch
        dimensions consisting of invertible matrices.

    Other Parameters
    ----------------
    out : tensor
        Output tensor. Ignored if `None`.

    Returns
    -------
    inv : tensor
        Tensor of shape `(*, n, n)` where `*` is zero or more batch
        dimensions consisting of invertible matrices.

    Raises
    ------
    RuntimeError
        If the matrix `A` or any matrix in the batch of matrices
        `A` is not invertible.

    """
    ...


def pinv(A, *, atol=None, rtol=None, hermitian=False, out=None):
    r"""
    Computes the pseudoinverse (Moore-Penrose inverse) of a matrix.

    The pseudoinverse may be defined algebraically but it is more
    computationally convenient to understand it through the SVD.

    !!! warning "`hermitian`"
        If `hermitian=True`, `A` is assumed to be Hermitian if complex or
        symmetric if real, but this is not checked internally. Instead,
        just the lower triangular part of the matrix is used in the
        computations.

    The singular values (or the norm of the eigenvalues when `hermitian=True`)
    that are below $\max(\text{atol}, \sigma_1 \cdot \text{rtol})$
    threshold are treated as zero and discarded in the computation, where
    $\sigma_1$ is the largest singular value (or eigenvalue).

    If `rtol` is not specified and `A` is a matrix of dimensions `(m, n)`,
    the relative tolerance is set to be $\text{rtol} = \max(m, n) \varepsilon$
    and $\varepsilon$ is the epsilon value for the dtype of `A`.
    If `rtol` is not specified and `atol` is specified to be larger than
    zero then `rtol` is set to zero.

    If `atol` or `rtol` is a torch.Tensor, its shape must be broadcastable
    to that of the singular values of `A` as returned by `linalg.svd`.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    !!! note
        This function uses `linalg.svd` if `hermitian=False` and
        `linalg.eigh` if `hermitian=True`.

    !!! note "Device"
        When inputs are on a CUDA device, this function synchronizes
        that device with the CPU.

    !!! note
        Consider using `linalg.lstsq` if possible for multiplying a
        matrix on the left by the pseudoinverse, as:
        ```python
        linalg.lstsq(A, B).solution == A.pinv() @ B
        ```
        It is always preferred to use `lstsq` when possible, as it is
        faster and more numerically stable than computing the pseudoinverse
        explicitly.

    !!! warning
        This function uses internally `linalg.svd` (or `linalg.eigh` when
        `hermitian=True`), so its derivative has the same problems as
        those of these functions. See the warnings in `linalg.svd` and
        `linalg.eigh` for more details.

    !!! info "See also"
        * `linalg.inv` computes the inverse of a square matrix.
        * `linalg.lstsq` computes `A.pinv() @ B` with a numerically stable
          algorithm.

    History
    -------
    !!! added "1.11"
        `rconv` was replaced with `atol` and `rtol` in torch version `1.11`.
    !!! added "1.8"
        `torch.linalg.pinv` was introduced in torch version `1.8`. Its
        signature was `pinv(input, rcond=1e-15, hermitian=False, *, out=None)`

    Parameters
    ----------
    A : tensor
        Tensor of shape (*, m, n) where * is zero or more batch dimensions.

    Other Parameters
    ----------------
    atol : float or tensor
        The absolute tolerance value. When `None` it's considered to be zero.
    rtol : float or tensor
        The relative tolerance value.
        See above for the value it takes when `None`.
    hermitian : bool
        Indicates whether `A` is Hermitian if complex or symmetric if real.
    out : tensor
        Output tensor. Ignored if `None`.

    Returns
    -------
    pinv : tensor
        Tensor of shape (*, n, m) where * is zero or more batch dimensions.
    """
    ...


def matrix_exp(A):
    r"""
    Computes the matrix exponential of a square matrix.

    Letting $\mathbb{K}$ be $\mathbb{R}$ or $\mathbb{C}$, this function
    computes the **matrix exponential** of $A \in \mathbb{K}^{n \times n}$,
    which is defined as

    $$\text{matrix_exp}(A) = \sum_{k=0}^\infty \frac{1}{k!} A^k
    \in \mathbb{K}^{n \times n}.$$

    If the matrix $A$ has eigenvalues $\lambda_i \in \mathbb{C}$, the matrix
    $\text{matrix_exp}(A)$ has eigenvalues $e^{\lambda_i}\in\mathbb{C}$.

    !!! note "Data types"
        Supports input of bfloat16, float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    History
    -------
    !!! added "1.11"
        `torch.linalg.matrix_exp` was introduced in torch version `1.11`.

    Parameters
    ----------
    A : tensor
        Tensor of shape `(*, m, m)` where `*` is zero or more batch
        dimensions.

    Returns
    -------
    expA : tensor
        Tensor of shape `(*, m, m)` where `*` is zero or more batch
        dimensions.

    """
    ...


def matrix_power(A, n, *, out=None):
    r"""
    Computes the n-th power of a square matrix for an integer n.

    If `n=0`, it returns the identity matrix (or batch) of the same
    shape as `A`. If `n` is negative, it returns the inverse of each
    matrix (if invertible) raised to the power of `abs(n)`.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of matrices, and if A is a batch of matrices
        then the output has the same batch dimensions.

    !!! note
        Consider using [`linalg.solve`][torchrelay.multivers.linalg.solve]
        if possible for multiplying a matrix on the left by a negative power
        as, if `n>0`:
        ```python
        matrix_power(torch.linalg.solve(A, B), n) == matrix_power(A, -n)  @ B
        ```
        It is always preferred to use
        [`linalg.solve`][torchrelay.multivers.linalg.solve] when possible,
        as it is faster and more numerically stable than computing $A^{-n}$
        explicitly.

    !!! info "See also"
        [`linalg.solve`][torchrelay.multivers.linalg.solve]
        computes `A.inverse() @ B` with a numerically stable algorithm.

    History
    -------
    !!! added "1.9"
        `torch.linalg.matrix_power` was introduced in torch version `1.9`.

    Parameters
    ----------
    A : tensor
        Tensor of shape `(*, m, m)` where `*` is zero or more batch
        dimensions.
    n : int
        The exponent.

    Other Parameters
    ----------------
    out : tensor
        Output tensor. Ignored if `None`.

    Returns
    -------
    An : tensor
        Tensor of shape `(*, m, m)` where `*` is zero or more batch
        dimensions.

    Raises
    ------
    RuntimeError
        If `n<0` and the matrix `A` or any matrix in the batch of
        matrices `A` is not invertible.
    """
    ...


def cross(input, other, *, dim=-1, out=None):
    r"""
    Computes the cross product of two 3-dimensional vectors.

    !!! note "Data types"
        Supports input of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        Also supports batches of vectors, for which it computes the
        product along the dimension dim. In this case, the output has
        the same batch dimensions as the inputs broadcast to a common shape.

    History
    -------
    !!! added "1.11"
        `torch.linalg.cross` was introduced in torch version `1.11`.
        Its default value for `dim` differs from `torch.cross`.
    !!! added "0.1"
        `torch.cross` was introduced in torch version `0.1`.

    Parameters
    ----------
    input : tensor
        The first input tensor.
    other : tensor
        The second input tensor.
    dim : int
        The dimension along which to take the cross-product.

    Other Parameters
    ----------------
    out : tensor
        The output tensor. Ignored if `None`.

    Returns
    -------
    out : tenor
        The output tensor.

    Raises
    ------
    RuntimeError
        If after broadcasting `input.size(dim) != 3` or
        `other.size(dim) != 3`.
    """
    ...


def matmul(input, other, *, out=None):
    r"""
    Matrix product of two tensors.

    The behavior depends on the dimensionality of the tensors as follows:

    * If both tensors are 1-dimensional, the dot product (scalar) is returned.
    * If both arguments are 2-dimensional, the matrix-matrix product is
      returned.
    * If the first argument is 1-dimensional and the second argument is
      2-dimensional, a 1 is prepended to its dimension for the purpose of
      the matrix multiply. After the matrix multiply, the prepended dimension
      is removed.
    * If the first argument is 2-dimensional and the second argument is
      1-dimensional, the matrix-vector product is returned.
    * If both arguments are at least 1-dimensional and at least one argument
      is N-dimensional (where N > 2), then a batched matrix multiply is
      returned. If the first argument is 1-dimensional, a 1 is prepended
      to its dimension for the purpose of the batched matrix multiply and
      removed after. If the second argument is 1-dimensional, a 1 is
      appended to its dimension for the purpose of the batched matrix
      multiple and removed after. The non-matrix (i.e. batch) dimensions
      are broadcasted (and thus must be broadcastable).
      For example, if `input` is a $(j \times 1 \times n \times n)$
      tensor and `other` is a $(k n \times n)$ tensor, `out` will be a
      $(j \times k \times n \times n)$ tensor. <br />
      Note that the broadcasting logic only looks at the batch dimensions
      when determining if the inputs are broadcastable, and not the matrix
      dimensions. For example, if `input` is a
      $(j \times 1 \times n \times m)$ tensor and other is a
      $(k \times m \times p)$ tensor, these inputs are valid for
      broadcasting even though the final two dimensions (i.e. the matrix
      dimensions) are different. `out` will be a
      $(j \times k \times n \times p)$ tensor.

    !!! warning
        The 1-dimensional dot product version of this function does not
        support an `out` parameter.

    History
    -------
    !!! added "1.10"
        `torch.linalg.matmul` was introduced in torch version `1.10`.
    !!! added "0.1"
        `torch.matmul` was introduced in torch version `0.1`.

    Parameters
    ----------
    input : tensor
        The first tensor to be multiplied
    other : tensor
        The second tensor to be multiplied

    Other Parameters
    ----------------
    out : tensor
        The output tensor.

    Returns
    -------
    out : tensor
        The output tensor.
    """
    ...


def vecdot(x, y, *, dim=-1, out=None):
    r"""
    Computes the dot product of two batches of vectors along a dimension.

    In symbols, this function computes

    $$\sum_{i=1}^n \bar{x}_i y_i .$$

    over the dimension `dim` where $\bar{x}_i$ denotes the conjugate for
    complex vectors, and it is the identity for real vectors.

    !!! note "Data types"
        Supports input of half, bfloat16, float, double, cfloat, cdouble
        and integral dtypes.

    !!! note "Dimensions"
        It also supports broadcasting.

    History
    -------
    !!! added "1.13"
        `torch.linalg.vecdot` was introduced in torch version `1.13`.

    Parameters
    ----------
    input : tensor
        First batch of vectors of shape `(*, n)`.
    other : tensor
        Second batch of vectors of shape `(*, n)`.

    Other Parameters
    ----------------
    dim : int
        Dimension along which to compute the dot product.
    out : tensor
        The output tensor.  Ignored if `None`.

    Returns
    -------
    out : tensor
        Output tensor of shape `(*)`.
    """
    ...


def multi_dot(tensors, *, out=None):
    r"""
    Efficiently multiplies two or more matrices by reordering the
    multiplications so that the fewest arithmetic operations are performed.

    !!! note "Data types"
        Supports inputs of float, double, cfloat and cdouble dtypes.

    !!! note "Dimensions"
        This function does not support batched inputs.

        Every tensor in tensors must be 2D, except for the first and last
        which may be 1D. If the first tensor is a 1D vector of shape `(n,)`
        it is treated as a row vector of shape `(1, n)`, similarly if the
        last tensor is a 1D vector of shape `(n,)` it is treated as a column
        vector of shape `(n, 1)`.

    If the first and last tensors are matrices, the output will be a matrix.
    However, if either is a 1D vector, then the output will be a 1D vector.

    !!! note "Differences with `numpy.linalg.multi_dot`"
        Unlike `numpy.linalg.multi_dot`, the first and last tensors must
        either be 1D or 2D whereas NumPy allows them to be nD.

    !!! warning
        This function does not broadcast.

    !!! note
        This function is implemented by chaining `torch.mm` calls after
        computing the optimal matrix multiplication order.

    History
    -------
    !!! added "1.9"
        `torch.linalg.multi_dot` was introduced and `torch.chain_matmul`
        was deprecated in torch version `1.9`.
        Keyword `out` was also added to `torch.chain_matmul`.
    !!! added "1.0"
        `torch.chain_matmul` was introduced and in torch version `1.0`.

    Parameters
    ----------
    tensors : sequence[tensor]
        Two or more tensors to multiply. The first and last tensors may
        be 1D or 2D. Every other tensor must be 2D.

    Other Parameters
    ----------------
    out : tensor
        Output tensor. Ignored if `None`.

    Returns
    -------
    out : tensor
        Output tensor.
    """
    ...


def vander(x, *, N=None):
    r"""
    Generates a Vandermonde matrix.

    Returns the Vandermonde matrix $V$

    $$V = \begin{pmatrix}
    1 & x_1 & x_1^2 & \cdots & x_1^{N-1} \\
    1 & x_2 & x_2^2 & \cdots & x_2^{N-1} \\
    1 & x_3 & x_3^2 & \cdots & x_3^{N-1} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    1 & x_n & x_n^2 & \cdots & x_n^{N-1}
    \end{pmatrix}$$

    for `N > 1`. If `N=None`, then `N = x.size(-1)` so that the output
    is a square matrix.

    !!! note "Data types"
        Supports inputs of float, double, cfloat, cdouble, and integral
        dtypes.

    !!! note "Dimensions"
        Also supports batches of vectors, and if x is a batch of vectors
        then the output has the same batch dimensions.

    !!! note "Differences with `numpy.vander`"
        Unlike `numpy.vander`, this function returns the powers of `x`
        in ascending order. To get them in the reverse order call
        `linalg.vander(x, N).flip(-1)`.

    History
    -------
    !!! added "1.12"
        `torch.linalg.vander` was introduced in torch version `1.12`.
    !!! added "1.6"
        `torch.vander` was introduced and in torch version `1.6`.
        Its signature was `torch.vander(x, N=None, increasing=False)`.

    Parameters
    ----------
    x : tensor
        Tensor of shape `(*, n)` where `*` is zero or more batch dimensions
        consisting of vectors.

    Other Parameters
    ----------------
    N : int
        Number of columns in the output. Default: `x.size(-1)`.

    Returns
    -------
    V : tensor
        Tensor of shape `(*, n, N)`.
    """
    ...
