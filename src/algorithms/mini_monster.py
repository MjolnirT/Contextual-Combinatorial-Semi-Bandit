import numpy as np

from Policy import EnumerationPolicy, ClassificationPolicy, RegressionPolicy
import Argmax

from .base import Semibandit


class MiniMonster(Semibandit):
    """
    Implementation of MiniMonster with a scikit_learn learning algorithm as the AMO.
    """
    def __init__(self, B, learning_alg = None, classification=True):
        self.B = B
        self.learning_alg = learning_alg
        if learning_alg == None:
            assert 'Pi' in dir(B), "No learning algorithm but simulator has no policies!"
            self.policy_type = EnumerationPolicy
            self.learning_alg = "enumerate"
        elif classification:
            assert B.L == 1, "Cannot use classification reduction for semibandits"
            self.policy_type = ClassificationPolicy
        else:
            self.policy_type = RegressionPolicy


    def init(self, T, params={}):
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

        self.reward = []
        self.opt_reward = []
        self.weights = []
        self.T = T
        self.action = None
        self.imp_weights = None
        self.t = 1
        self.leader = None
        self.history = []
        self.num_amo_calls = 0
        if 'mu' in params.keys():
            self.mu = params['mu']
        else:
            self.mu = 1.0
        self.num_unif = 0
        self.num_leader = 0

    def update(self, x, A, y_vec, r):
        full_rvec = np.zeros(x.get_K())
        full_rvec[A] = y_vec
        self.history.append((x, A, full_rvec, 1.0/self.imp_weights))
        if self.t >= 10 and self.t in self.training_points:
            if self.verbose:
                print("---- Training ----", flush=True)
            pi = Argmax.argmax2(self.B, self.history, policy_type=self.policy_type, learning_alg = self.learning_alg)
            self.num_amo_calls += 1
            self.leader = pi
            self.weights = self._solve_op()
            if self.verbose:
                print("t: ", self.t, " Support: ", len(self.weights), flush=True)

        self.action = None
        self.imp_weights = None
        self.t += 1

    def get_action(self, x):
        """
        Choose a composite action for context x.

        Implements projection, smoothing, mixing etc.
        Computes importance weights and stores in self.imp_weights.
        """
        ## Compute slate and base action distributions
        p = {}
        p2 = np.zeros(x.get_K())
        for item in self.weights:
            pi = item[0]
            w = item[1]
            (p, p2) = self._update_dists(p, p2, pi.get_action(x), w)

        ## Mix in leader
        if self.leader != None:
            (p, p2) = self._update_dists(p, p2, self.leader.get_action(x), 1 - np.sum([z[1] for z in self.weights]))
        ## Compute importance weights by mixing in uniform
        p2 = (1-self._get_mu())*p2 + (self._get_mu())*x.get_L()/x.get_K()
        self.imp_weights = p2

        ## Decide what action to play
        unif = np.random.binomial(1, self._get_mu())
        if unif or self.leader is None:
            ## Pick a slate uniformly at random
            act = np.random.choice(x.get_K(), size=x.get_L(), replace=False)
            self.num_unif += 1
        else:
            ## Pick a slate from the policy distribution.
            p = [(k,v) for (k,v) in p.items()]
            draw = np.random.multinomial(1, [x[1] for x in p])
            act = self._unhash_list(p[np.where(draw)[0][0]][0])
            if p[np.where(draw)[0][0]][1] > 0.5:
                self.num_leader += 1
        self.action = act
        return act

    def _update_dists(self, slate_dist, base_dist, slate, weight):
        """
        This is a subroutine for the projection step.
        Update the slate_dist and base_dist distributions by
        incorporating slate played with weight.
        """
        key = self._hash_list(slate)
        if key in slate_dist.keys():
            slate_dist[key] = slate_dist[key] + weight
        else:
            slate_dist[key] = weight
        for a in slate:
            base_dist[a] += weight
        return (slate_dist, base_dist)

    def _get_mu(self):
        """
        Return the current value of mu_t
        """
        a = self.mu
        b = self.mu*np.sqrt(self.B.K)/np.sqrt(self.B.L*self.t)
        c = np.min([a,b])
        return np.min([1,c])

    def _hash_list(self, lst):
        """
        There is a hashing trick we are using to make a dictionary of
        composite actions. _hash_list and _unhash_list implement
        the hash and inverse hash functions.

        The hash is to write the composite action as a number in base K.
        """
        return np.sum([lst[i]*self.B.K**i for i in range(len(lst))])

    def _unhash_list(self, num):
        lst = []
        if num == 0:
            return np.array([0 for i in range(self.B.L)])
        for i in range(self.B.L):
            rem = num % self.B.K
            lst.append(rem)
            num = (num-rem)//self.B.K
        return lst

    def _solve_op(self):
        """
        Main optimization logic for MiniMonster.
        """
        H = self.history
        mu = self._get_mu()
        Q = []
        psi = 1

        ## MEMOIZATION
        ## 1. Unnormalized historical reward for each policy in supp(Q)
        ## 2. Feature matrix on historical contexts
        ## 3. Recommendations for each policy in supp(Q) for each context in history.
        predictions = {}
        ## vstack the features once here. This is a caching optimization
        if self.policy_type == ClassificationPolicy:
            features = np.zeros((1, H[0][0].get_dim()))
            for x in H:
                features = np.vstack((features, x[0].get_features()))
            features = features[1:,:]
        elif self.policy_type == RegressionPolicy:
            features = np.zeros((1, H[0][0].get_ld_dim()))
            for x in H:
                features = np.vstack((features, x[0].get_ld_features()))
            features = features[1:,:]
        else:
            features = None

        ## Invariant is that leader is non-Null
        (leader_reward, predictions) = self._get_reward(H, self.leader, predictions, features=features)
        leader_reward = leader_reward

        q_rewards = {}
        for item in Q:
            pi = item[0]
            (tmp,predictions) = self._get_reward(H, pi, predictions, features=features)
            q_rewards[pi] = tmp

        updated = True
        iterations = 0
        while updated and iterations < 20:
            iterations += 1
            updated = False
            ## First IF statement
            score = np.sum([x[1]*(2*self.B.K*self.B.L/self.B.L + self.B.K*(leader_reward - q_rewards[x[0]])/(psi*self.t*self.B.L*mu)) for x in Q])
            if score > 2*self.B.K*self.B.L/self.B.L:
                c = (2*self.B.K*self.B.L/self.B.L)/score
                Q = [(x[0], c*x[1]) for x in Q]
                updated = True

            ## argmax call and coordinate descent update.
            ## Prepare dataset
            Rpi_dataset = []
            Vpi_dataset = []
            Spi_dataset = []
            for i in range(self.t):
                context = H[i][0]
                q = self._marginalize(Q, context, predictions)
                act = np.arange(context.get_K())
                r1 = 1.0/(self.t*q)
                r2 = self.B.K/(self.t*psi*mu*self.B.L)*H[i][2]/H[i][3]
                r3 = 1.0/(self.t*q**2)
                weight = np.ones(context.get_K())

                Vpi_dataset.append((context, act, r1, weight))
                Rpi_dataset.append((context, act, r2, weight))
                Spi_dataset.append((context, act, r3, weight))
            dataset = Rpi_dataset
            dataset.extend(Vpi_dataset)

            ## AMO call
            pi = Argmax.argmax2(self.B, dataset, policy_type=self.policy_type, learning_alg = self.learning_alg)
            self.num_amo_calls += 1

            ## This is mostly to make sure we have the predictions cached for this new policy
            if pi not in q_rewards.keys():
                (tmp,predictions) = self._get_reward(H, pi, predictions, features=features)
                q_rewards[pi] = tmp
                if q_rewards[pi] > leader_reward:
                    self.leader = pi
                    leader_reward = q_rewards[pi]

            assert pi in predictions.keys(), "Uncached predictions for new policy pi"
            ## Test if we need to update
            (Dpi,predictions) = self._get_reward(dataset, pi, predictions)
            target = 2*self.B.K*self.B.L/self.B.L + self.B.K*leader_reward/(psi*self.t*mu*self.B.L)
            if Dpi > target:
                ## Update
                updated = True
                Dpi = Dpi - (2*self.B.K*self.B.L/self.B.L + self.B.K*leader_reward/(psi*self.t*mu*self.B.L))
                (Vpi,ptwo) = self._get_reward(Vpi_dataset, pi, predictions)
                (Spi,ptwo) = self._get_reward(Spi_dataset, pi, predictions)
                toadd = (Vpi + Dpi)/(2*(1-mu)*Spi)
                Q.append((pi, toadd))
        return Q

    def _get_reward(self, dataset, pi, predictions, features=None):
        """
        For a policy pi whose predictions are cached in predictions dict,
        compute the cumulative reward on dataset.
        """
        if pi not in predictions.keys():
            ## This is going to go horribly wrong if dataset is not the right size
            assert len(dataset) == self.t, "If predictions not yet cached, dataset should have len = self.t"
            predictions[pi] = dict(zip([y[0].get_name() for y in dataset], pi.get_all_actions([y[0] for y in dataset], features=features)))
        score = 0.0
        for item in dataset:
            x = item[0].get_name()
            r = item[2]
            w = item[3]
            score += np.sum(r[predictions[pi][x]]*w[predictions[pi][x]])

        return (score, predictions)

    def _marginalize(self, Q, x, predictions):
        """
        Marginalize a set of weights Q for context x
        using the predictions cache.
        """
        p = np.zeros(self.B.K, dtype=np.longdouble)
        for item in Q:
            pi = item[0]
            w = item[1]
            p[predictions[pi][x.get_name()]] += w
        p = (1.0-self._get_mu())*p + (self._get_mu())*float(self.B.L)/float(self.B.K)
        return p
