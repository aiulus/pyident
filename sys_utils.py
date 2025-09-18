import numpy as np
from scipy import linalg as sla

def c2d(A,B,dt):
    n,m = B.shape
    M = np.block([[A, B],[np.zeros((m, n+m))]])
    E = sla.expm(M*dt)
    Ad = E[:n,:n]; Bd = E[:n, n:n+m]
    return Ad, Bd

def simulate(T, x0, Ad, Bd, u):
    X = np.zeros((Ad.shape[0], T+1))
    X[:,0] = x0
    for t in range(T):
        X[:,t+1] = Ad @ X[:,t] + Bd @ u[:,t]
    return X