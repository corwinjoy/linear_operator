#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import BlockBLockLinearOperator
from linear_operator.test.base_test_case import BaseTestCase
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestBlockBlockSimple(BaseTestCase, unittest.TestCase):
    def test_multiply(self):
        T = 2
        N = 4
        M = 3
        K = 5

        A = torch.randn(T, T, N, M)
        B = torch.randn(T, T, M, K)

        A_blo = BlockBLockLinearOperator.from_tensor(A, T)
        B_blo = BlockBLockLinearOperator.from_tensor(B, T)
        res = A_blo._matmul(B_blo)
        res_dense = res.to_dense()

        expected = A.permute(0, 2, 1, 3).reshape(T * N, T * M) @ B.permute(0, 2, 1, 3).reshape(T * M, T * K)
        self.assertAllClose(res_dense, expected)


class TestBlockBlockLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = False
    T = 2
    N = 4
    M = 3
    K = 5

    A = torch.randn(T, T, N, M)
    B = torch.randn(T, T, M, K)

    def create_linear_op(self):
        A_blo = BlockBLockLinearOperator.from_tensor(self.A, self.T)
        return A_blo

    def evaluate_linear_op(self, linear_op):
        D = linear_op.to_dense()
        return D


if __name__ == "__main__":
    unittest.main()
