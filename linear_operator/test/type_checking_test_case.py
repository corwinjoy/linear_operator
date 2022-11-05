# These are tests to directly check torchtyping signatures as extended to LinearOperator.
# The idea is to verify that dimension tests are working as expected.

import unittest
from typing import Union

import torch
from torchtyping import TensorType  # type: ignore
from torchtyping import patch_typeguard
from torchtyping.tensor_type import TensorTypeMixin  # type: ignore
from typeguard import typechecked

from linear_operator.operators import DenseLinearOperator, LinearOperator

# For flake8, matrix dimensions
M = None
C = None
N = None

patch_typeguard()  # use before @typechecked


class MetaLinearOperatorType(type(LinearOperator), type(TensorTypeMixin)):
    pass


# Inherit from LinearOperator so that IDEs are happy to find methods on functions
# annotated as LinearOperatorType.
class LinearOperatorType(LinearOperator, TensorTypeMixin, metaclass=MetaLinearOperatorType):
    base_cls = LinearOperator


@typechecked
def linop_matmul_fn(
    lo: LinearOperatorType[..., "M", "N"], vec: Union[TensorType[..., "N", "C"], TensorType["N"]]
) -> Union[TensorType[..., "M", "C"], TensorType["M"]]:
    r"""
    Performs a matrix multiplication :math:`\mathbf KM` with the (... x M x N) matrix :math:`\mathbf K`
    that lo represents. Should behave as
    :func:`torch.matmul`. If the LinearOperator represents a batch of
    matrices, this method should therefore operate in batch mode as well.

    :param lo: the K = MxN left hand matrix
    :param vec: the matrix :math:`\mathbf M` to multiply with (... x N x C).
    :return: :math:`\mathbf K \mathbf M` (... x M x C)
    """

    res = lo.matmul(vec)
    return res


@typechecked
def linop_matmul_fn_bad_lo(
    lo: LinearOperatorType["N"], vec: Union[TensorType[..., "N", "C"], TensorType["N"]]
) -> Union[TensorType[..., "M", "C"], TensorType["M"]]:
    r"""
    As above, but with bad size array for lo
    """

    res = lo.matmul(vec)
    return res


@typechecked
def linop_matmul_fn_bad_vec(
    lo: LinearOperatorType[..., "M", "N"], vec: TensorType[..., "N", "C"]
) -> Union[TensorType[..., "M", "C"], TensorType["M"]]:
    r"""
    Performs a matrix multiplication :math:`\mathbf KM` with the (... x M x N) matrix :math:`\mathbf K`
    that lo represents. Should behave as
    :func:`torch.matmul`. If the LinearOperator represents a batch of
    matrices, this method should therefore operate in batch mode as well.

    :param lo: the K = MxN left hand matrix
    :param vec: the matrix :math:`\mathbf M` to multiply with (... x N x C).
    :return: :math:`\mathbf K \mathbf M` (... x M x C)
    """

    res = lo.matmul(vec)
    return res


@typechecked
def linop_matmul_fn_bad_retn(
    lo: LinearOperatorType[..., "M", "N"], vec: Union[TensorType[..., "N", "C"], TensorType["N"]]
) -> TensorType[..., "M", "C"]:
    r"""
    As above, but with bad return size
    """

    res = lo.matmul(vec)
    return res


class TestTypeChecking(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mat = torch.tensor([[3, -1, 0], [-1, 3, 0], [0, 0, 3]], dtype=torch.float)
        self.vec = torch.randn(3)
        self.lo = DenseLinearOperator(mat)

    def test_linop_matmul_fn(self):
        linop_matmul_fn(self.lo, self.vec)

    def test_linop_matmul_fn_bad_lo(self):
        with self.assertRaises(TypeError):
            linop_matmul_fn_bad_lo(self.lo, self.vec)

    def test_linop_matmul_fn_bad_vec(self):
        with self.assertRaises(TypeError):
            linop_matmul_fn_bad_vec(self.lo, self.vec)

    def test_linop_matmul_fn_bad_retn(self):
        with self.assertRaises(TypeError):
            linop_matmul_fn_bad_retn(self.lo, self.vec)


if __name__ == "__main__":
    unittest.main()
