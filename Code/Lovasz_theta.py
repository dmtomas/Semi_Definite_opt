import numpy as np
import cvxpy as cp
import scipy.linalg as sp
"""
edges = [(0, 1),
         (0, 3),
         (0, 5),
         (1, 2),
         (1, 4),
         (2, 3),
         (2, 3),
         (3, 4),
         (4, 5)]
"""
edges = [(0, 2),
         (0, 3),
         (1, 4),
         (1, 3),
         (2, 4)]
n = 6
a = cp.Variable((n, 1))

M = cp.Variable((n, n), symmetric=True)


temp = [[0.0 for i in range(n + 1)] for i in range(n + 1)]
temp[0][0] = 1
for i in range(1, n+1):
    temp[0][i] = M[i-1][i-1]
    temp[i][0] = M[i-1][i-1]
for i in range(n):
    for j in range(n):
        temp[i+1][j+1] = M[i][j]

delta = cp.bmat(temp)

constraints = [M[i][j] == 0 for (i,j) in edges]
constraints += [delta >> 0]

objective = cp.trace(M)
prob = cp.Problem(cp.Maximize(objective), constraints).solve()

print(prob)