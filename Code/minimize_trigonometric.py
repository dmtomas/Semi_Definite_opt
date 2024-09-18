import numpy as np
import cvxpy as cp

alpha = cp.Variable()
beta = cp.Variable()
gamma = cp.Variable()
delta = cp.Variable()

X = cp.bmat([[1 , alpha, beta], [alpha, gamma, delta], [beta, delta, 1-gamma]])

constraints = [X >> 0]
constraints += [X[1][1] + X[2][2] == 1]
constraints += [alpha <= 1]
constraints += [beta <= 1]
constraints += [gamma <= 1]
constraints += [delta <= 1]
constraints += [alpha >= -1]
constraints += [beta >= -1]
constraints += [gamma >= -1]
constraints += [delta >= -1]


objective = alpha + beta + gamma + delta
prob = cp.Problem(cp.Maximize(objective), constraints).solve()
print(f"The maximum is {prob}")

print(f"The value should be {np.arccos(X.value[0][1])}")
print(f"This is another possibility {np.arcsin(X.value[0][2])}")