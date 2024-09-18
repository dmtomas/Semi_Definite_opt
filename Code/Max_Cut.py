import numpy as np
import cvxpy as cp
import scipy.linalg as sp

edges = [(0, 1),
         (0, 3),
         (0, 5),
         (1, 2),
         (1, 4),
         (2, 3),
         (2, 3),
         (3, 4),
         (4, 5)]

X = cp.Variable((6, 6), symmetric=True)
constraints = [X >> 0]
constraints += [X[i,i] == 1 for i in range(6)]


objective = sum(0.5 * (1 - X[i, j]) for (i, j) in edges)
prob = cp.Problem(cp.Maximize(objective), constraints)

prob.solve()

x = sp.sqrtm(X.value)
u = np.random.randn(6)
x = np.sign(x @ u)
print(x)
print(X.value)