import numpy as np
import sklearn.linear_model
import Argmax

from Policy import EnumerationPolicy, RegressionPolicy
from .base import Semibandit


class EELS(Semibandit):
    """
    Implementation of GLM-Semibandit with scikit_learn learning algorithm as the AMO.
    """
    def __init__(self, B):
        self.B = B
        self.learning_alg = "enumerate"
        self.link = "linear"

    def init(self, T, params=None):
        if 'learning_alg' in params.keys():
            self.learning_alg = params['learning_alg']
        if 'link' in params.keys():
            self.link = params['link']
        if self.learning_alg == 'enumerate':
            assert 'Pi' in dir(self.B), "No learning algorithm but simulator has no policies!"
            self.policy_type = EnumerationPolicy
        else:
            self.policy_type = RegressionPolicy

        self.lambda_star = np.sqrt(self.B.K*T)                                     ## TODO: maybe needs to change
        self.t_star = (T*self.B.K/self.B.L)**(2.0/3)*(1.0/self.B.L)**(1.0/3)       ## TODO: maybe needs to change
        self.reward = []
        self.opt_reward = []
        self.weights = None
        self.cov = np.matrix(np.zeros((self.B.L, self.B.L)))
        self.reg_data = []
        self.T = T
        self.action = None
        self.imp_weights = None
        self.t = 1
        self.history = []
        self.emp_best = None

        self.reward = []
        self.opt_reward = []

    def get_action(self, x):
        if self.weights is None:
            act = np.random.choice(x.get_K(), size=x.get_L(), replace=False)
            self.action = act
        else:
            self.action = self.emp_best.get_weighted_action(x,self.weights)
        return self.action

    def update(self, x, act, y_vec, r):
        if self.emp_best != None:
            return
        ## Otherwise we have to update
        vec = y_vec
        full_rvec = np.zeros(x.get_K())
        full_rvec[act] = y_vec*(self.B.L/self.B.K)*np.ones(len(act))
        self.history.append((x,full_rvec))
        self.cov += np.matrix(vec).T*np.matrix(vec)
        self.reg_data.append((np.matrix(vec).T, r))
        [u,v] = np.linalg.eig(self.cov)
        if np.min(np.real(u)) > self.lambda_star and self.t > self.t_star:
            if self.verbose:
                print("---- Training ----", flush=True)
            if self.link == "linear":
                model = sklearn.linear_model.LinearRegression(fit_intercept=True)
                X = np.matrix(np.hstack([self.reg_data[i][0] for i in range(len(self.reg_data))])).T
                y = np.hstack([self.reg_data[i][1] for i in range(len(self.reg_data))])
                model.fit(X,y)
                self.weights = np.array(model.coef_)[0,:]
            if self.link == "logistic":
                model = sklearn.linear_model.LogisticRegression(C=1000.0,fit_intercept=True)
                X = np.matrix(np.hstack([self.reg_data[i][0] for i in range(len(self.reg_data))])).T
                y = np.hstack([self.reg_data[i][1] for i in range(len(self.reg_data))])
                model.fit(X,y)
                self.weights = np.array(model.coef_)[0,:]
            self.emp_best = Argmax.weighted_argmax(self.B,self.history,self.weights, link=self.link, policy_type=self.policy_type, learning_alg=self.learning_alg)
        self.t += 1
