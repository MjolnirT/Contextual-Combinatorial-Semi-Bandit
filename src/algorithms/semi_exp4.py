import numpy as np

from Util import mixture_decomp

from .base import Semibandit


class SemiExp4(Semibandit):
    """
    Semibandit EXP4 as in Kale, Reyzin, Schapire.
    Uses parameters prescribed in the paper.

    This only works for Semibandit Simulators where there
    is a finite enumerable policy class. To get this to
    work on the Learning-to-Rank datasets, you need to construct
    such a policy class before hand and make it accessible to the
    SemibanditSim object.

    Specifically the algorithm uses B.Pi, which is a multidimensional
    array for which B.Pi[policy, context, :] is the prescribed composite
    action. The SemibanditSim object needs to have this structure.
    """

    def __init__(self, B):
        assert "Pi" in dir(B), "Cannot run EXP4 without explicit policy class"
        self.B = B
        self.link = "linear"

    def init(self, T, params={}):
        self.reward = []
        self.opt_reward = []
        self.dist = [1.0/self.B.N for i in range(self.B.N)]
        self.T = T
        ## Default parameter settings from the paper
        if 'gamma' in params.keys():
            self.gamma = params['gamma']
        else:
            self.gamma = np.sqrt((self.B.K/self.B.L)*np.log(self.B.N)/self.T)
        if 'eta' in params.keys():
            self.eta = params['eta']
        else:
            self.eta = np.sqrt((1-self.gamma)*self.B.L*np.log(self.B.N)/(self.B.K*self.T))
        self.act_dist = []
        self.action = None
        self.weight = np.ones(self.B.L)
        if 'weight' in params.keys():
            self.weight = params['weight']
        if 'link' in params.keys():
            self.link = params['link']

    def update(self, x, act, y_vec, r):
        """
        This step enumerates the policy class.
        """
        reward = np.zeros(self.B.K)
        reward[act] = y_vec/(self.B.L*self.act_dist[act])

        for n in range(self.B.N):
            if self.link == "linear":
                policy_reward = np.dot(self.weight,reward[self.B.Pi[n].get_action(x)])
            if self.link == "logistic":
                policy_reward = 1.0/(1+np.exp(-np.dot(self.weight, reward[self.B.Pi[n].get_action(x)])))
            self.dist[n] = self.dist[n]*np.exp(self.eta*policy_reward)
        self.dist = self.dist/np.sum(self.dist)

    def get_action(self,x):
        """
        This also enumerates the policy class.
        """
        p = np.zeros(self.B.K)
        for n in range(self.B.N):
            p[self.B.Pi[n].get_action(x)] += self.dist[n]/self.B.L
        p = (1-self.gamma)*p + self.gamma/self.B.K*np.ones(self.B.K)
        self.act_dist = p
        (M, z) = mixture_decomp(p, self.B.L)
        samp = np.random.multinomial(1, z)
        row = M[np.where(samp)[0][0],:]
        self.action = np.where(row)[0]
        return(self.action)
