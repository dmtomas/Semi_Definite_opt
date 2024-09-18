import cvxpy as cp
import numpy as np

n = 3
A = cp.Variable((n,n), symmetric=True)
C = [[0, 0, 0], [1, 0, 0], [0, 0, 0]]
constraints = [A >> 0]
constraints += [A[0, 0] == 1]
constraints += [A[1, 1] == 1]
constraints += [A[2, 2] == 1]
constraints += [A[0, 1] == A[0, 2]]
constraints += [A[0, 1] == A[1, 2]]


prob = cp.Problem(cp.Minimize(cp.trace(C @ A)),
                  constraints)
prob.solve()

print("The optimal value is", prob.value)
print("A solution X is")
print(A.value)

# Otra forma de escribir lo mismo

alpha = cp.Variable()
A = cp.bmat([[1, alpha, alpha], [alpha, 1, alpha], [alpha, alpha, 1]])
prob = cp.Problem(cp.Minimize(alpha), [A>>0]).solve()
print("The optimal value is", prob)