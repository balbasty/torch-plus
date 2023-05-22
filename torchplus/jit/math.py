import torch
from torchplus import torch_version


@torch.jit.script
def square(x):
    return x * x


@torch.jit.script
def square_(x):
    return x.mul_(x)


@torch.jit.script
def cube(x):
    return x * x * x


@torch.jit.script
def cube_(x):
    return square_(x).mul_(x)


@torch.jit.script
def pow4(x):
    return square(square(x))


@torch.jit.script
def pow4_(x):
    return square_(square_(x))


@torch.jit.script
def pow5(x):
    return x * pow4(x)


@torch.jit.script
def pow5_(x):
    return pow4_(x).mul_(x)


@torch.jit.script
def pow6(x):
    return square(cube(x))


@torch.jit.script
def pow6_(x):
    return square_(cube_(x))


@torch.jit.script
def pow7(x):
    return pow6(x) * x


@torch.jit.script
def pow7_(x):
    return pow6_(x).mul_(x)


# floor_divide returns wrong results for negative values, because it truncates
# instead of performing a proper floor. In recent version of pytorch, it is
# advised to use div(..., rounding_mode='trunc'|'floor') instead.
# Here, we only use floor_divide on positive values so we do not care.
if torch_version('>=', (1, 8)):

    @torch.jit.script
    def floor_div(x, y) -> torch.Tensor:
        return torch.div(x, y, rounding_mode='trunc')

    @torch.jit.script
    def floor_div_int(x, y: int) -> torch.Tensor:
        return torch.div(x, y, rounding_mode='trunc')

    @torch.jit.script
    def trunc_div(x, y) -> torch.Tensor:
        return torch.div(x, y, rounding_mode='trunc')

    @torch.jit.script
    def trunc_div_int(x, y: int) -> torch.Tensor:
        return torch.div(x, y, rounding_mode='trunc')

else:

    @torch.jit.script
    def floor_div(x, y) -> torch.Tensor:
        return (x / y).floor().to(x.dtype)

    @torch.jit.script
    def floor_div_int(x, y: int) -> torch.Tensor:
        return (x / y).floor().to(x.dtype)

    @torch.jit.script
    def trunc_div(x, y) -> torch.Tensor:
        int_dtype = torch.long if x.is_floating_point() else x.dtype
        return (x / y).to(int_dtype).to(x.dtype)

    @torch.jit.script
    def trunc_div_int(x, y: int) -> torch.Tensor:
        int_dtype = torch.long if x.is_floating_point() else x.dtype
        return (x / y).to(int_dtype).to(x.dtype)
