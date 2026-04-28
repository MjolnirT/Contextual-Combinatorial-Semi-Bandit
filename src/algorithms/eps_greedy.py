import numpy as np

from Policy import EnumerationPolicy, ClassificationPolicy, RegressionPolicy
import Argmax

from .base import Semibandit


class EpsGreedy(Semibandit):
    """
    Epsilon Greedy algorithm for semibandit learning.
    Can use scikit_learn LearningAlg as oracle

    Current implementation uses a constant value for epsilon.
    """
    def __init__(self, B, learning_alg="enumerate", classification=False):
        """
        Initialize the epsilon greedy algorithm.

        Args:
        B -- Semibandit Simulator
        learning_alg -- scikit learn regression algorithm.
                        Should support fit() and predict() methods.
        """
        self.B = B
        self.link = "linear"
        self.learning_alg = learning_alg
        if learning_alg == "enumerate":
            assert 'Pi' in dir(B), "No learning algorithm but simulator has no policies!"
            self.policy_type = EnumerationPolicy
            self.learning_alg = "enumerate"
        elif classification:
            assert B.L == 1, "Cannot use classification reduction for semibandits"
            self.policy_type = ClassificationPolicy
        else:
            self.policy_type = RegressionPolicy

    def init(self, T, params={}):
        """
        Initialize the current run.
        The EpsGreedy algorithm maintains a lot more state.

        Args:
        T -- Number of rounds of interaction.
        """
        if "eps" in params.keys():
            self.eps = params['eps']
        else:
            self.eps = 0.1
        if "reward" in params.keys() and params['reward'] == True:
            self.use_reward_features = False
        else:
            self.use_reward_features = True
            if 'weight' in params.keys():
                self.weights = params['weight']
            elif self.B.L == 1:
                self.weights=np.array([1.0])
            else:
                self.weights = self.B.weight

        if 'train_all' in params.keys() and params['train_all']:
            self.train_all = True
        else:
            self.train_all = False

        if 'learning_alg' in params.keys():
            self.learning_alg = params['learning_alg']
        if self.learning_alg == "enumerate":
            assert 'Pi' in dir(self.B), "No learning algorithm but simulator has no policies!"
            self.policy_type = EnumerationPolicy
        elif 'classification' in params.keys():
            assert B.L == 1, "Cannot use classification reduction for semibandits"
            self.policy_type = ClassificationPolicy
        else:
            self.policy_type = RegressionPolicy

        if "link" in params.keys():
            self.link = params['link']

        self.training_points = []
        i = 4
        ## Default training schedule is exponential
        while True:
            self.training_points.append(int(np.sqrt(2)**i))
            i+=1
            if np.sqrt(2)**i > T:
                break
        if 'schedule' in params.keys():
            if params['schedule'] == 'lin':
                ## Switch to linear training schedule with step 100
                self.training_points = [50]
                i=1
                while True:
                    self.training_points.append(int(i*100))
                    i+=1
                    if i*100>T:
                        break
        print(self.training_points)

        self.reward = []
        self.opt_reward = []
        self.T = T
        self.t = 1
        self.action = None
        self.leader = None
        self.history = []
        self.ber = 0
        self.num_unif = 0

    def get_action(self, x):
        """
        Select a composite action to play for this context.
        Also updates the importance weights.

        Args:
        x -- context.
        """
        self.imp_weights = (x.get_L()/x.get_K())*np.ones(x.get_K())
        if self.leader is not None and self.train_all:
            self.imp_weights = (self._get_eps())*self.imp_weights
            self.imp_weights[self.leader.get_action(x)] += (1-self._get_eps())
        self.ber = np.random.binomial(1,np.min([1, self._get_eps()]))
        if self.leader is None or self.ber:
            A = np.random.choice(x.get_K(), size=x.get_L(), replace=False)
            self.action = A
            self.num_unif += 1
        elif self.use_reward_features:
            A = self.leader.get_weighted_action(x,self.weights)
        else:
            A = self.leader.get_action(x)
        self.action = A
        return A

    def update(self, x, act, y_vec, r):
        """
        Update the state for the algorithm.
        We currently call the AMO whenever t is a perfect square.

        Args:
        x -- context.
        act -- composite action (np.array of length L)
        r_vec -- reward vector (np.array of length K)
        """
        if self.use_reward_features:
            full_rvec = np.zeros(self.B.K)
            full_rvec[act] = y_vec
            if self.train_all or self.ber:
                self.history.append((x, act, full_rvec, 1.0/self.imp_weights))
        elif self.ber:
            self.history.append((x,act,r))
        if self.t >= 10 and self.t in self.training_points:
            if self.verbose:
                print("----Training----", flush=True)
            if self.use_reward_features:
                self.leader = Argmax.argmax2(self.B, self.history, policy_type = self.policy_type, learning_alg = self.learning_alg)
            else:
                self.leader = Argmax.reward_argmax(self.B, self.history, policy_type = self.policy_type, learning_alg = self.learning_alg)
        self.t += 1

    def _get_eps(self):
        """
        Return the current value of epsilon.
        """
        return np.max([1.0/np.sqrt(self.t), self.eps])
