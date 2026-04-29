"""
Entry point for the Semibandit experiments previously buried in
src/Semibandits.py's __main__ block.

Run from the repo root, e.g.:
    python run_semibandits.py --T 30000 --dataset mslr30k --L 3 --I 0 \
        --alg mini --learning_alg gb2 --param 0.01
"""
import os
import sys
import time
import argparse

import numpy as np
import sklearn.ensemble
import sklearn.tree
import sklearn.linear_model

# Make the modules under src/ importable (they use flat imports like
# `import Simulators`, `from Policy import *`, etc.).
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Simulators
import settings

from algorithms import EpsGreedy, LinUCB, MiniMonster, SquareCB, RandomPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', action='store', default=1000,
                        help='number of rounds', type=int)
    parser.add_argument('--dataset', action='store',
                        choices=['synth', 'mq2007', 'mq2008', 'yahoo',
                                 'mslr', 'mslrsmall', 'mslr30k', 'xor'])
    parser.add_argument('--L', action='store', default=5, type=int)
    parser.add_argument('--I', action='store', default=0, type=int)
    parser.add_argument('--noise', action='store', default=None)
    parser.add_argument('--alg', action='store', default='all',
                        choices=['mini', 'eps', 'lin', 'sqcb', 'random'])
    parser.add_argument('--learning_alg', action='store', default=None,
                        choices=[None, 'gb2', 'gb5', 'tree', 'lin'])
    parser.add_argument('--param', action='store', default=None)
    parser.add_argument('--wandb', action='store_true',
                        help='log per-round SquareCB metrics to wandb '
                             '(entity/project from src/wandb_logger.py)')
    parser.add_argument('--wandb_tag', action='store', default=None,
                        help='optional wandb tag for grouping runs')
    parser.add_argument('--log_every', action='store', default=1, type=int,
                        help='log every Nth round to wandb (default 1)')
    parser.add_argument('--opt_against', action='store', default=None,
                        choices=[None, 'oracle'],
                        help="comparator for opt_reward: None uses the "
                             "simulator's get_best_reward() (typically the "
                             "best fixed policy Pi*); 'oracle' uses the "
                             "top-L hindsight oracle sum(top-L of curr_r).")
    return parser.parse_args()


def make_learning_alg(name):
    if name == "gb2":
        return lambda: sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=2)
    if name == "gb5":
        return lambda: sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=5)
    if name == "tree":
        return lambda: sklearn.tree.DecisionTreeRegressor(max_depth=2)
    if name == "lin":
        return lambda: sklearn.linear_model.LinearRegression()
    return None


def main():
    Args = parse_args()
    print(Args, flush=True)
    if Args.noise is not None:
        Args.noise = float(Args.noise)
        outdir = './results/%s_T=%d_L=%d_e=%0.1f/' % (Args.dataset, Args.T, Args.L, Args.noise)
    else:
        outdir = './results/%s_T=%d_L=%d_e=0.0/' % (Args.dataset, Args.T, Args.L)
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    if Args.param is not None:
        Args.param = float(Args.param)

    loop = True
    if Args.dataset in ('mslr', 'mslrsmall', 'mslr30k', 'yahoo'):
        loop = False

    B = Simulators.DatasetBandit(dataset=Args.dataset, L=Args.L, loop=loop, metric=None, noise=Args.noise)
    Bval = None
    if Args.dataset != 'yahoo':
        Bval = Simulators.DatasetBandit(dataset=Args.dataset, L=Args.L, loop=False, metric=None, noise=Args.noise)

    if Args.dataset == 'mslr30k' and Args.I < 20:
        order = np.load(settings.DATA_DIR + "mslr/mslr30k_train_%d.npz" % (Args.I))
        print("Setting order for Iteration %d" % (Args.I), flush=True)
        o = order['order']
        B.contexts.order = o[np.where(o < B.contexts.X)[0]]
        B.contexts.curr_idx = 0
    if Args.dataset == 'yahoo' and Args.I < 20:
        order = np.load(settings.DATA_DIR + "yahoo/yahoo_train_%d.npz" % (Args.I))
        print("Setting order for Iteration %d" % (Args.I), flush=True)
        o = order['order']
        B.contexts.order = o[np.where(o < B.contexts.X)[0]]
        B.contexts.curr_idx = 0

    print("Setting seed for Iteration %d" % (Args.I), flush=True)
    B.set_seed(Args.I)

    learning_alg = make_learning_alg(Args.learning_alg)

    logger = None
    if Args.wandb:
        from wandb_logger import init_run
        logger = init_run(Args, tag=Args.wandb_tag)

    def attach_logger(params):
        if logger is not None:
            params['logger'] = logger
        if Args.opt_against is not None:
            params['opt_against'] = Args.opt_against
        return params

    if Args.alg == "lin":
        L = LinUCB(B)
        if Args.param is not None:
            if os.path.isfile(outdir + "lin_%0.5f_rewards_%d.out" % (Args.param, Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r, reg, val_tmp) = L.play(Args.T, verbose=True, validate=Bval,
                                       params=attach_logger({'delta': Args.param}))
            stop = time.time()
            np.savetxt(outdir + "lin_%0.5f_rewards_%d.out" % (Args.param, Args.I), r)
            np.savetxt(outdir + "lin_%0.5f_validation_%d.out" % (Args.param, Args.I), val_tmp)
            np.savetxt(outdir + "lin_%0.5f_time_%d.out" % (Args.param, Args.I), np.array([stop - start]))
        else:
            if os.path.isfile(outdir + "lin_default_rewards_%d.out" % (Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r, reg, val_tmp) = L.play(Args.T, verbose=True, validate=Bval,
                                       params=attach_logger({}))
            stop = time.time()
            np.savetxt(outdir + "lin_default_rewards_%d.out" % (Args.I), r)
            np.savetxt(outdir + "lin_default_validation_%d.out" % (Args.I), val_tmp)
            np.savetxt(outdir + "lin_default_time_%d.out" % (Args.I), np.array([stop - start]))

    if Args.alg == "mini":
        if learning_alg is None:
            print("Cannot run MiniMonster without learning algorithm")
            sys.exit(1)
        M = MiniMonster(B, learning_alg=learning_alg, classification=False)
        if Args.param is not None:
            if os.path.isfile(outdir + "mini_%s_%0.3f_rewards_%d.out" % (Args.learning_alg, Args.param, Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r, reg, val_tmp) = M.play(Args.T, verbose=True, validate=Bval,
                                       params=attach_logger({'weight': np.ones(Args.L), 'mu': Args.param}))
            stop = time.time()
            np.savetxt(outdir + "mini_%s_%0.3f_rewards_%d.out" % (Args.learning_alg, Args.param, Args.I), r)
            np.savetxt(outdir + "mini_%s_%0.3f_validation_%d.out" % (Args.learning_alg, Args.param, Args.I), val_tmp)
            np.savetxt(outdir + "mini_%s_%0.3f_time_%d.out" % (Args.learning_alg, Args.param, Args.I), np.array([stop - start]))
        else:
            if os.path.isfile(outdir + "mini_%s_default_rewards_%d.out" % (Args.learning_alg, Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r, reg, val_tmp) = M.play(Args.T, verbose=True, validate=Bval,
                                       params=attach_logger({'weight': np.ones(Args.L)}))
            stop = time.time()
            np.savetxt(outdir + "mini_%s_default_rewards_%d.out" % (Args.learning_alg, Args.I), r)
            np.savetxt(outdir + "mini_%s_default_validation_%d.out" % (Args.learning_alg, Args.I), val_tmp)
            np.savetxt(outdir + "mini_%s_default_time_%d.out" % (Args.learning_alg, Args.I), np.array([stop - start]))

    if Args.alg == "eps":
        if learning_alg is None:
            print("Cannot run EpsGreedy without learning algorithm")
            sys.exit(1)
        E = EpsGreedy(B, learning_alg=learning_alg, classification=False)
        if Args.param is not None:
            if os.path.isfile(outdir + "epsall_%s_%0.3f_rewards_%d.out" % (Args.learning_alg, Args.param, Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r, reg, val_tmp) = E.play(Args.T, verbose=True, validate=Bval,
                                       params=attach_logger({'weight': np.ones(Args.L), 'eps': Args.param, 'train_all': True}))
            stop = time.time()
            np.savetxt(outdir + "epsall_%s_%0.3f_rewards_%d.out" % (Args.learning_alg, Args.param, Args.I), r)
            np.savetxt(outdir + "epsall_%s_%0.3f_validation_%d.out" % (Args.learning_alg, Args.param, Args.I), val_tmp)
            np.savetxt(outdir + "epsall_%s_%0.3f_time_%d.out" % (Args.learning_alg, Args.param, Args.I), np.array([stop - start]))
        else:
            if os.path.isfile(outdir + "epsall_%s_default_rewards_%d.out" % (Args.learning_alg, Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r, reg, val_tmp) = E.play(Args.T, verbose=True, validate=Bval,
                                       params=attach_logger({'weight': np.ones(Args.L), 'train_all': True}))
            stop = time.time()
            np.savetxt(outdir + "epsall_%s_default_rewards_%d.out" % (Args.learning_alg, Args.I), r)
            np.savetxt(outdir + "epsall_%s_default_validation_%d.out" % (Args.learning_alg, Args.I), val_tmp)
            np.savetxt(outdir + "epsall_%s_default_time_%d.out" % (Args.learning_alg, Args.I), np.array([stop - start]))

    if Args.alg == "sqcb":
        if learning_alg is None:
            print("Cannot run SquareCB without learning algorithm")
            sys.exit(1)
        S = SquareCB(B, learning_alg=learning_alg, classification=False)
        if Args.param is not None:
            if os.path.isfile(outdir + "sqcb_%s_%0.3f_rewards_%d.out" % (Args.learning_alg, Args.param, Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r, reg, val_tmp) = S.play(Args.T, verbose=True, validate=Bval,
                                       params=attach_logger({'gamma_multiplier': Args.param,
                                                             'seed': Args.I,
                                                             'log_every': Args.log_every}))
            stop = time.time()
            np.savetxt(outdir + "sqcb_%s_%0.3f_rewards_%d.out" % (Args.learning_alg, Args.param, Args.I), r)
            np.savetxt(outdir + "sqcb_%s_%0.3f_validation_%d.out" % (Args.learning_alg, Args.param, Args.I), val_tmp)
            np.savetxt(outdir + "sqcb_%s_%0.3f_time_%d.out" % (Args.learning_alg, Args.param, Args.I), np.array([stop - start]))
        else:
            if os.path.isfile(outdir + "sqcb_%s_default_rewards_%d.out" % (Args.learning_alg, Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r, reg, val_tmp) = S.play(Args.T, verbose=True, validate=Bval,
                                       params=attach_logger({'seed': Args.I,
                                                             'log_every': Args.log_every}))
            stop = time.time()
            np.savetxt(outdir + "sqcb_%s_default_rewards_%d.out" % (Args.learning_alg, Args.I), r)
            np.savetxt(outdir + "sqcb_%s_default_validation_%d.out" % (Args.learning_alg, Args.I), val_tmp)
            np.savetxt(outdir + "sqcb_%s_default_time_%d.out" % (Args.learning_alg, Args.I), np.array([stop - start]))

    if Args.alg == "random":
        # The base Semibandit class is the uniform-random slate baseline:
        # get_action() draws a size-L subset uniformly at random, update() is a no-op.
        # No learner, no tunable param.
        if os.path.isfile(outdir + "random_rewards_%d.out" % (Args.I)):
            print('---- ALREADY DONE ----')
            sys.exit(0)
        R = RandomPolicy(B)
        start = time.time()
        (r, reg, val_tmp) = R.play(Args.T, verbose=True, validate=Bval,
                                   params=attach_logger({}))
        stop = time.time()
        np.savetxt(outdir + "random_rewards_%d.out" % (Args.I), r)
        np.savetxt(outdir + "random_validation_%d.out" % (Args.I), val_tmp)
        np.savetxt(outdir + "random_time_%d.out" % (Args.I), np.array([stop - start]))

    if logger is not None:
        logger.finish()

    print("---- DONE ----")


if __name__ == '__main__':
    main()
