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


@typechecked
def matmul_fn(
    mat: TensorType[..., "M", "N"], vec: Union[TensorType[..., "N", "C"], TensorType["N"]]
) -> Union[TensorType[..., "M", "C"], TensorType["M"]]:
    r"""
    Performs a matrix multiplication :math:`\mathbf KM` with the (... x M x N) matrix :math:`\mathbf K`
    that this LinearOperator represents. Should behave as
    :func:`torch.matmul`. If the LinearOperator represents a batch of
    matrices, this method should therefore operate in batch mode as well.

    ..note::
        This method is intended to be used only internally by various
        Functions that support backpropagation (e.g., :class:`Matmul`).
        Once this method is defined, it is strongly recommended that one
        use :func:`~linear_operator.LinearOperator.matmul` instead, which makes use of this
        method properly.

    :param mat: the K = MxN left hand matrix
    :param vec: the matrix :math:`\mathbf M` to multiply with (... x N x C).
    :return: :math:`\mathbf K \mathbf M` (... x M x C)
    """

    res = DenseLinearOperator(mat).matmul(vec)
    return res


def test_matmul_fn():
    print("running test_matmul_fn")
    mat = torch.tensor([[3, -1, 0], [-1, 3, 0], [0, 0, 3]], dtype=torch.float)
    vec = torch.randn(3)
    matmul_fn(mat, vec)
    print("completed test_matmul_fn")


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
    that this LinearOperator represents. Should behave as
    :func:`torch.matmul`. If the LinearOperator represents a batch of
    matrices, this method should therefore operate in batch mode as well.

    ..note::
        This method is intended to be used only internally by various
        Functions that support backpropagation (e.g., :class:`Matmul`).
        Once this method is defined, it is strongly recommended that one
        use :func:`~linear_operator.LinearOperator.matmul` instead, which makes use of this
        method properly.

    :param mat: the K = MxN left hand matrix
    :param vec: the matrix :math:`\mathbf M` to multiply with (... x N x C).
    :return: :math:`\mathbf K \mathbf M` (... x M x C)
    """

    res = lo.matmul(vec)
    return res


def test_lo_fn():
    print("running test_lo_fn")
    mat = torch.tensor([[3, -1, 0], [-1, 3, 0], [0, 0, 3]], dtype=torch.float)
    vec = torch.randn(3)
    lo = DenseLinearOperator(mat)
    linop_matmul_fn(lo, vec)
    print("completed test_lo_fn")


test_matmul_fn()
test_lo_fn()
