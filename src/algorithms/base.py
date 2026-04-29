import numpy as np


class Semibandit(object):
    """
    Default semibandit learning algorithm.  Provides interface for
    semibandit learning but also implements the random policy that
    picks random slates on each iteration and does no updating.
    """
    def __init__(self, B):
        """
        Args:
        B -- A SemibanditSim instance.
        """
        self.B = B

    def init(self, T, params={}):
        """
        Initialize the semibandit algorithm for a run.
        Args:
        T -- maximum number of rounds.
        """
        self.reward = []
        self.opt_reward = []
        self.dist = [1.0/self.B.N for i in range(self.B.N)]

    def play(self, T, params={}, verbose=True, validate=None, loss=False):
        """
        Execute this algorithm on the semibandit simulator for T rounds.

        Returns:
        cumulative reward (np.array),
        cumulative optimal reward (np.array)
        cumulative regret (np.array)
        """
        self.verbose=verbose
        self.init(T, params=params)
        self.val_scores = []
        logger = params.get('logger') if isinstance(params, dict) else None
        opt_against = params.get('opt_against') if isinstance(params, dict) else None
        log_every = int(params.get('log_every', 1)) if isinstance(params, dict) else 1
        cum_simple = 0.0
        cum_opt = 0.0
        for t in range(T):
            if t != 0 and np.log2(t+1) == int(np.log2(t+1)) and verbose:
                print("t = %d, r = %0.3f, ave_regret = %0.3f" % (t, np.cumsum(self.reward)[len(self.reward)-1], (np.cumsum(self.opt_reward) - np.cumsum(self.reward))[len(self.reward)-1]/(t+1)), flush=True)
            if validate != None and t != 0 and t % 500 == 0:
                val = validate.offline_evaluate(self, train=False)
                if verbose:
                    print("t=%d: val = %0.3f" % (t, val))
                self.val_scores.append(val)
            x = self.B.get_new_context()
            if x == None:
                break
            p = self.get_action(x)
            r = self.B.get_slate_reward(p)
            if opt_against == 'oracle':
                cr = np.asarray(self.B.curr_r, dtype=np.float64).ravel()
                L = x.get_L()
                o = float(np.sort(cr)[-L:].sum())
            else:
                o = self.B.get_best_reward()
            self.reward.append(r)
            self.opt_reward.append(o)
            if logger is not None and getattr(logger, 'active', False):
                simple_r = float(np.sum(r))
                opt_r = float(np.sum(o))
                cum_simple += simple_r
                cum_opt += opt_r
                if (t + 1) % log_every == 0:
                    logger.log({
                        'reward/simple': simple_r,
                        'reward/avg_cum': cum_simple / (t + 1),
                        'reward/optimal': opt_r,
                        'regret/simple': opt_r - simple_r,
                        'regret/avg_cum': (cum_opt - cum_simple) / (t + 1),
                    }, step=t + 1)
            if loss:
                self.update(x, p, [1.0-a for a in self.B.get_base_rewards(p)], self.B.get_slate_reward(p))
            else:
                self.update(x, p, self.B.get_base_rewards(p), self.B.get_slate_reward(p))
            l1 = np.cumsum(self.reward)
        l2 = np.cumsum(self.opt_reward)
        return (l1[9::10], (l2-l1)[9::10], self.val_scores)

    def update(self, x, a, y_vec, r):
        """
        Update the state of the semibandit algorithm with the most recent
        context, composite action, and reward.

        The default semibandit doesn't do any updating.

        Args:
        x -- context (should be hashable)
        a -- composite action (np.array of length L)
        r -- reward vector (np.array of length K)
        """
        pass

    def get_action(self, x):
        """
        Pick a composite action to play for this context.

        Args:
        x -- context

        Returns:
        Composite action (np.array of length L)
        """
        act = np.random.choice(x.get_K(), size=x.get_L(), replace=False)
        self.action = act
        return self.action
