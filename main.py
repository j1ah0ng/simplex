#!/usr/bin/env python3
import numpy as np


class Simplex:
    def __init__(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c

        ### Number of variables
        self.n = c.size
        n = self.n
        ### Number of constraints
        self.m = A.shape[0]
        m = self.m

        if self.n != A.shape[1]:
            raise ValueError

        # Form the simplex tableau where
        # x1 x2 ... xn s1 s2 ... sm z
        self.T = np.zeros((m + 1, n + m + 1))
        self.T[:m, :n] = A
        self.T[:-1, n:-1] = np.eye(m)
        self.T[-1, :n] = -1 * c
        self.T[-1, -1] = 1

        self.T = np.hstack((self.T, np.vstack((b, np.array([0])))))
        print(self.T)

    def is_optimal(self):
        return np.all(self.T[-1, :-2] >= 0)

    def find_pivot(self):
        return np.argmin(self.T[-1, :-2])

    def find_row(self, var_idx):
        divs = np.divide(self.T[:-1, -1], self.T[:-1, var_idx])
        divs[self.T[:-1, var_idx] <= 0] = np.inf
        return np.argmin(divs)

    def zero(self):
        self.T[np.isclose(self.T, 0.0)] = 0

    def eliminate(self, row, col):
        coef = self.T[row, col]
        self.T[row, :] = self.T[row, :] / coef
        for r in range(self.T.shape[0]):
            if r == row:
                continue
            ratio = self.T[r, col]
            self.T[r, :] = self.T[r, :] - (self.T[row, :] * ratio)
        self.zero()

    def solve(self):
        while not self.is_optimal():
            col = self.find_pivot()
            row = self.find_row(col)
            self.eliminate(row, col)
