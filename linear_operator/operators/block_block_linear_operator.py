from typing import Union

import torch
from torch import Tensor

from ._linear_operator import LinearOperator


class BlockBLockLinearOperator(LinearOperator):
    def __init__(self, blocks: Tensor) -> None:
        assert len(blocks.size()) > 2, "must have nested list"
        assert blocks.size(-3) == blocks.size(-4), "must be square over block dimensions"

        if len(blocks.size()) > 4:
            blocks = blocks.squeeze()

        assert (len(blocks.size())) < 5, "batch dimensions are not supported"

        super().__init__(blocks)

        self.blocks = blocks

    def _matmul(self, other: Union[LinearOperator, torch.Tensor]) -> LinearOperator:
        T = self.blocks.size(0)
        if isinstance(other, self.__class__):
            # Check size is the same
            assert T == other.blocks.size(0)
            assert T == other.blocks.size(1)
            output_shape = torch.Size(self.blocks.size()[:3] + (other.blocks.size(3),))
            output = torch.zeros(output_shape)
            for i in range(T):
                for j in range(T):
                    out_ij = torch.zeros(self.blocks[0, 0].shape[0], other.blocks[0, 0].shape[1])
                    for k in range(T):
                        out_ij += self.blocks[i, k] @ other.blocks[k, j]
                    output[i, j] = out_ij
            return self.__class__(output)
        elif isinstance(other, Tensor):
            # Check both matrix dims divisible by T,
            # reshape to (T, T, ), call block multiplication
            if other.size(0) % T == 0 and other.size(1) % T == 0:
                other_blocks = self.from_2D(other, T)
                other_op = BlockBLockLinearOperator(other_blocks)
                return self._matmul(other_op)

        A = self.to_dense()
        B = other.to_dense()
        res = A @ B
        return res

    def to_dense(self) -> Tensor:
        return self.to_2D()

    def to_2D(self) -> Tensor:
        T = self.blocks.size(0)
        N = self.blocks.size(-2)
        M = self.blocks.size(-1)
        blocks_dense = self.blocks.permute(0, 2, 1, 3).reshape(T * N, T * M)
        return blocks_dense

    @staticmethod
    def from_2D(A: Tensor, T: int) -> Tensor:
        N = A.size(0) // T
        M = A.size(1) // T
        A_blocks = A.reshape(T, N, T, M).permute(0, 2, 1, 3)
        return A_blocks

    def _size(self):
        sz = self.blocks[0, 0].size()
        T = self.blocks.size(0)
        return torch.Size((T * sz[0], T * sz[1]))

    @property
    def matrix_shape(self) -> torch.Size:
        return self._size()

    def _diag(self):
        T = self.blocks.size(0)
        out = []
        for i in range(T):
            diagonal = self.blocks[i, i].diagonal()
            out.append(diagonal)
        return torch.concat(out)

    def diagonal(self, offset: int = 0, dim1: int = -2, dim2: int = -1) -> torch.Tensor:
        return self._diag()

    def add_diagonal(self, diag: torch.Tensor) -> LinearOperator:
        r"""
        Adds an element to the diagonal of the matrix.

        :param diag: Diagonal to add
        :return: :math:`\mathbf A + \text{diag}(\mathbf d)`, where :math:`\mathbf A` is the linear operator
            and :math:`\mathbf d` is the diagonal component
        """

        diag_shape = diag.shape

        # Standard case: we have a different entry for each diagonal element
        if len(diag_shape) and diag_shape[-1] != 1:
            # We need to get the target batch shape, and expand the diag_tensor to the appropriate size
            # If we do not, there will be issues with backpropagating gradients
            expanded_diag = diag.expand(self.shape[:-1])
            T = self.blocks.size(0)
            blocks = self.blocks.clone()
            idx = 0
            for i in range(T):
                for j in range(min(blocks.size(-2), blocks.size(-1))):
                    blocks[i, i, j, j] += expanded_diag[idx]
                    idx += 1
            return self.__class__(blocks)

        # Other case: we are using broadcasting to define a constant entry for each diagonal element
        # In this case, we want to exploit the structure

        val = diag.item()
        T = self.blocks.size(0)
        blocks = self.blocks.clone()
        for i in range(T):
            for j in range(min(blocks.size(-2), blocks.size(-1))):
                blocks[i, i, j, j] += val
        return self.__class__(blocks)

    def _transpose_nonbatch(self):
        return self  # Diagonal matrices are symmetric

    @classmethod
    def from_tensor(cls, tensor: Tensor, T: int = 1):
        if len(tensor.size()) == 2:
            blocks = cls.from_2D(tensor, T)
            return cls(blocks)
        return cls(tensor)
