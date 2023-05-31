from typing import List, Optional, Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor

from ._linear_operator import IndexType, LinearOperator
from .dense_linear_operator import DenseLinearOperator
from .zero_linear_operator import ZeroLinearOperator


class BlockTensorLinearOperator(LinearOperator):
    def __init__(self, linear_operators: List[List[LinearOperator]]) -> None:
        assert isinstance(linear_operators, list)
        assert len(linear_operators) > 0, "must have non-empty list"
        assert len(linear_operators[0]) == len(linear_operators), "must be square over block dimensions"

        super().__init__(linear_operators)

        self.linear_operators = linear_operators
        self.num_tasks = len(self.linear_operators)
        self.block_rows = linear_operators[0][0].shape[0]
        self.block_cols = linear_operators[0][0].shape[1]

    @staticmethod
    def square_ops(T):
        """Return an empty (square) list of operators of shape TxT"""
        ops = []
        for i in range(T):
            tmp = []
            for j in range(T):
                tmp.append([])
            ops.append(tmp)
        return ops

    def _matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[torch.Tensor, "*batch2 N C"], Float[torch.Tensor, "*batch2 N"]],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:

        T = self.num_tasks

        # A is block [N * T1, M * T2] and B is block [O * S1, P * S2]. If A and B have conformal block counts
        # ie T2==S1 as well as M==O then use the blockwise algorithm. Else use to_dense()
        if isinstance(rhs, self.__class__) and self.num_tasks == rhs.num_tasks and self.block_cols == rhs.block_rows:
            output = BlockTensorLinearOperator.square_ops(T)
            for i in range(T):
                for j in range(T):
                    out_ij = self.linear_operators[i][0] @ rhs.linear_operators[0][j]
                    for k in range(1, T):
                        out_ij += self.linear_operators[i][k] @ rhs.linear_operators[k][j]
                    output[i][j] = out_ij
            return self.__class__(output)
        elif isinstance(rhs, Tensor) and rhs.ndim == 2:
            # Check both matrix dims divisible by T,
            # reshape to (T, T, ), call block multiplication
            if rhs.size(0) % T == 0 and rhs.size(1) % T == 0:
                # A is block [N * T, M * T] and B is a general tensor/operator of shape [O, P].
                # If O and P are both divisible by T,
                # then interpret B as a [O//T * T, P//T * T] block matrix
                O_T = rhs.size(0) // T
                P_T = rhs.size(1) // T
                rhs_blocks_raw = rhs.reshape(T, O_T, T, P_T)
                rhs_blocks = rhs_blocks_raw.permute(0, 2, 1, 3)
                rhs_op = BlockTensorLinearOperator.from_tensor(rhs_blocks, T)
                return self._matmul(rhs_op)

        # Failover implementation. Convert to dense and multiply matricies
        A = self.to_dense()
        B = rhs.to_dense()
        res = A @ B
        return res

    def matmul(
        self: Float[LinearOperator, "*batch M N"],
        other: Union[Float[Tensor, "*batch2 N P"], Float[Tensor, "*batch2 N"], Float[LinearOperator, "*batch2 N P"]],
    ) -> Union[Float[Tensor, "... M P"], Float[Tensor, "... M"], Float[LinearOperator, "... M P"]]:
        return self._matmul(other)

    def to_dense(self: Float[LinearOperator, "*batch M N"]) -> Float[Tensor, "*batch M N"]:
        out = []
        for i in range(self.num_tasks):
            rows = []
            for j in range(self.num_tasks):
                rows.append(self.linear_operators[i][j].to_dense())
            out.append(torch.concat(rows, axis=1))
        return torch.concat(out, axis=0)

    def _size(self) -> torch.Size:
        sz = self.linear_operators[0][0].size()
        return torch.Size([self.num_tasks * sz[0], self.num_tasks * sz[1]])

    @property
    def dtype(self) -> Optional[torch.dtype]:
        return self.linear_operators[0][0].dtype

    @property
    def device(self) -> Optional[torch.device]:
        return self.linear_operators[0][0].device

    def representation(self) -> Tuple[torch.Tensor, ...]:
        """
        Returns the Tensors that are used to define the LinearOperator
        """
        representation = []
        for op_row in self.linear_operators:
            for op in op_row:
                representation += tuple(op.representation())
        return tuple(representation)

    def _diag(self):
        out = []
        for i in range(self.num_tasks):
            diagonal = self.linear_operators[i][i].diagonal()
            out.append(diagonal)
        return torch.concat(out, axis=1)

    def _transpose_nonbatch(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch N M"]:
        out = []
        for i in range(self.num_tasks):
            rows = []
            for j in range(self.num_tasks):
                rows.append(self.linear_operators[j][i].mT)
            out.append(rows)
        return BlockTensorLinearOperator(out)

    def _getitem(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> LinearOperator:
        # Perform the __getitem__
        # TODO make this faster, see block_linear_operator
        tsr = self.to_dense()
        res = tsr[(*batch_indices, row_index, col_index)]
        return DenseLinearOperator(res)

    @classmethod
    def from_tensor(cls, tensor: Tensor, num_tasks: int):
        def tensor_to_linear_op(t):
            if torch.count_nonzero(t) > 0:
                return DenseLinearOperator(t)
            return ZeroLinearOperator(*t.size(), dtype=t.dtype, device=t.device)

        linear_ops = [
            [tensor_to_linear_op(t[0]) for t in list(torch.tensor_split(tensor[i], num_tasks))]
            for i in range(num_tasks)
        ]
        return cls(linear_ops)