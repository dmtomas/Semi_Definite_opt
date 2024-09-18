import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing

def Kennedy(beta, alpha):
    return - (0.5 * np.exp(-np.abs(alpha - beta)**2) + 0.5 * (1 - np.exp(-np.abs(alpha + beta)**2)))

def Optimal_Kennedy(alpha):
    bounds = [[0, 1]]
    value = dual_annealing(Kennedy, bounds=bounds, args=[alpha])
    while type(value.fun) != type(0.0):
        print("test")
        value = dual_annealing(Kennedy, bounds=bounds, args=[alpha])

    return - value.fun



n = 2
alpha = 0.2

def optimal(alpha):
    gamma = cp.Variable()
    beta = 0.65
    X = cp.bmat([[gamma * (1 + 2 * alpha * beta), 0], [0, gamma * (1 - 2 * alpha* beta)]])


    constraints = [X >> 0]
    constraints += [X[0][0] <= 1]
    constraints += [X[1][0] <= 1]
    constraints += [X[0][1] <= 1]
    constraints += [X[1][1] <= 1]
    objective = 0.5 * X[0][0] + 0.5 * (1 - X[1][1])

    prob = cp.Problem(cp.Maximize(objective), constraints).solve()
    #return np.sqrt(-np.log(X.value[1][1])) + alpha
    return np.sqrt(-np.log(gamma.value))

alphas = np.linspace(0.05, 1, 50)
numerical_beta = []
betas = []

for i in range(len(alphas)):
    betas.append(-Kennedy(optimal(alphas[i]), alphas[i]))
    numerical_beta.append(Optimal_Kennedy(alphas[i]))

plt.ylabel("beta")
plt.xlabel("x")
plt.plot(alphas, betas, label="semi definite")
plt.plot(alphas, numerical_beta, label="dual annealing")
plt.legend()
plt.show()