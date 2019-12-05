import numpy as np

def linear_kernel(x, y, **kwargs):
    return np.dot(x, y)

def polynomial_kernel(x, y, **kwargs):
    degree = kwargs['degree']
    gamma = kwargs['gamma']
    coef0 = kwargs['coef0']
    return (coef0 + gamma * np.dot(x, y)) ** degree

def gaussian_kernel(x, y, **kwargs):
    gamma = kwargs['gamma']
    return np.exp(-gamma * (np.linalg.norm(x-y)**2))
