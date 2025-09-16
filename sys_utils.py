import numpy as np

def c2d(A,B,dt):
    n,m = B.shape
    M = np.block([[A, B],[np.zeros((m, n+m))]])
    E = sla.expm(M*dt)
    Ad = E[:n,:n]; Bd = E[:n, n:n+m]
    return Ad, Bd