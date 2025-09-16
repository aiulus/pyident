import numpy as np
import numpy.linalg as npl

# Dynamic mode decomposition with control (DMDc)
class DMDC:
    name = "dmdc"
    def fit(self, X: np.ndarray, U:np.ndarray):
        X0, X1 = X[:, :-1], X[:, 1:]
        Theta = np.vstack([X0, U])
        AB = X1 @ np.linalg.pinv(Theta)
        n = X0.shape[0]
        self.A_hat = AB[:, :n]
        self.B_hat = AB[:, n:]
        return self
    
        def metrics(self, A_true: np.ndarray, B_true: np.ndarray):
            eA = npl.norm(self.A_hat - A_true, 'fro')
            eB = npl.norm(self.B_hat - B_true, 'fro')
            return dict(errA_fro=float(eA), 
                        errB_fro=float(eB), 
                        err_joint=float((eA*eA + eB*eB)**0.5))
