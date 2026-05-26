import numpy as np
import scipy.linalg

from .base import Semibandit


class LinUCB(Semibandit):
    """
    Implementation of Semibandit Linucb.
    This algorithm of course only works if features are available
    in the SemibanditSim object.

    It must also have a fixed dimension featurization and expose
    B.d as an instance variable.
    """
    def __init__(self, B):
        """
        Initialize a linUCB object.
        """
        self.B = B

    def init(self, T, params={}):
        """
        Initialize the regression target and the
        feature covariance.
        """
        self.T = T
        self.d = self.B.d
        self.b_vec = np.matrix(np.zeros((self.d,1)))
        self.cov = np.matrix(np.eye(self.d))
        self.Cinv = scipy.linalg.inv(self.cov)
        self.weights = self.Cinv*self.b_vec
        self.t = 1

        if "delta" in params.keys():
            self.delta = params['delta']
        else:
            self.delta = 0.05
        if "schedule" in params.keys():
            self.schedule = params['schedule']
        else:
            self.schedule = 100

        self.reward = []
        self.opt_reward = []

    def update(self, x, A, y_vec, r):
        """
        Update the regression target and feature cov.
        """
        features = np.matrix(x.get_ld_features())
        for i in range(len(A)):
            self.cov += features[A[i],:].T*features[A[i],:]
            self.b_vec += y_vec[i]*features[A[i],:].T

        self.t += 1
        if self.t % self.schedule == 0:
            self.Cinv = scipy.linalg.inv(self.cov)
            self.weights = self.Cinv*self.b_vec

    def get_action(self, x):
        """
        Find the UCBs for the predicted reward for each base action
        and play the composite action that maximizes the UCBs
        subject to whatever constraints.
        """
        features = np.matrix(x.get_ld_features())
        K = x.get_K()

        alpha = np.sqrt(self.d)*self.delta
        ucbs = [features[k,:]*self.weights + alpha*np.sqrt(features[k,:]*self.Cinv*features[k,:].T) for k in range(K)]

        ucbs = [a[0,0] for a in ucbs]
        ranks = np.argsort(ucbs)
        return ranks[K-self.B.L:K]

    def predict_slate_value(self, x, p):
        features = np.matrix(x.get_ld_features())
        preds = features[np.asarray(p), :] * self.weights
        return float(np.sum(preds))
