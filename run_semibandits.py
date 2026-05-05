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

from algorithms import EpsGreedy, LinUCB, MiniMonster, SquareCB, RegCB, RandomPolicy


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
                        choices=['mini', 'eps', 'lin', 'sqcb', 'regcb', 'random'])
    parser.add_argument('--learning_alg', action='store', default=None,
                        choices=[None, 'gb2', 'gb5', 'tree', 'lin', 'rff', 'nonlinear',
                                 'xgb', 'xgb2', 'xgb5'])
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
    parser.add_argument('--sampler', action='store', default='marginal',
                        choices=['marginal', 'bandit', 'depround'],
                        help="SquareCB IGW sampler: 'marginal' (per-arm "
                             "IGW marginals + Madow systematic, default), "
                             "'depround' (per-arm IGW marginals + dependent "
                             "rounding), or 'bandit' (IGW over C(K,L) "
                             "macro-actions via rejection sampling).")
    parser.add_argument('--y_scale', action='store', default=1.0, type=float,
                        help="SquareCB target scaling: regressor trains on "
                             "y/y_scale and predicts in original-y units. "
                             "Use 4.0 for relevance datasets where y is in "
                             "{0..4} (default 1.0 — no scaling).")
    # XGBoost-oracle hyperparams (forwarded only when set; otherwise the
    # oracle's own defaults apply). Surfaced so the tuning runs in
    # src/SQUARECB_XGBOOST_TUNING.md can be reproduced from the CLI.
    parser.add_argument('--xgb_max_depth', type=int, default=None)
    parser.add_argument('--xgb_learning_rate', type=float, default=None)
    parser.add_argument('--xgb_min_child_weight', type=float, default=None)
    parser.add_argument('--xgb_fit_every', type=int, default=None)
    parser.add_argument('--xgb_warmup', type=int, default=None)
    parser.add_argument('--xgb_num_boost_round', type=int, default=None)
    parser.add_argument('--xgb_buffer_size', type=int, default=None)
    parser.add_argument('--xgb_reg_lambda', type=float, default=None)
    parser.add_argument('--xgb_reg_alpha', type=float, default=None)
    # RegCB-specific knobs (per src/REGCB_IMPLEMENTATION_PLAN.md). RegCB is
    # restricted to L=1; --param is the confidence slack ``beta``.
    parser.add_argument('--regcb_mode', choices=['optimistic', 'elimination'],
                        default='optimistic')
    parser.add_argument('--regcb_warmup', type=int, default=50)
    parser.add_argument('--regcb_binsearch_steps', type=int, default=20)
    parser.add_argument('--regcb_synth_weight', type=float, default=1.0)
    parser.add_argument('--regcb_reward_min', type=float, default=0.0)
    parser.add_argument('--regcb_reward_max', type=float, default=4.0)
    parser.add_argument('--regcb_ridge_alpha', type=float, default=1.0)
    parser.add_argument('--regcb_gb_n_estimators', type=int, default=100)
    parser.add_argument('--regcb_refit_per_step', action='store_true',
                        help="Use the eager paper-faithful binary search "
                             "(refit oracle every round per arm per binsearch "
                             "step). Default: lazy FALCON-style refit at "
                             "exponential epochs only.")
    return parser.parse_args()


def _xgb_overrides(Args):
    """Collect XGBoost-oracle kwargs explicitly set on the CLI."""
    keys = ('max_depth', 'learning_rate', 'min_child_weight',
            'fit_every', 'warmup', 'num_boost_round',
            'buffer_size', 'reg_lambda', 'reg_alpha')
    return {k: v for k in keys
            if (v := getattr(Args, 'xgb_' + k)) is not None}


def make_learning_alg(name):
    if name == "gb2":
        return lambda: sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=2)
    if name == "gb5":
        return lambda: sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=5)
    if name == "tree":
        return lambda: sklearn.tree.DecisionTreeRegressor(max_depth=2)
    if name == "lin":
        return lambda: sklearn.linear_model.LinearRegression()
    if name in ("rff", "nonlinear", "xgb", "xgb5", "xgb2"):
        # SquareCB-only model-type selectors; legacy learning_alg gate
        # in the sqcb branch needs a non-None marker, but the actual
        # regressor factory is not used (online oracle takes over).
        return name
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

    B = Simulators.DatasetBandit(dataset=Args.dataset, L=Args.L, loop=True, metric=None, noise=Args.noise)
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

    state = {'logger': None}

    def attach_logger(params):
        # Lazy init so the wandb run is created with the algorithm's
        # actual hyperparams (e.g. gamma_multiplier, mu, eps, delta) in
        # wandb.config, not just the bare CLI args.
        if Args.wandb and state['logger'] is None:
            from wandb_logger import init_run
            state['logger'] = init_run(Args, params, tag=Args.wandb_tag, B=B)
        if state['logger'] is not None:
            params['logger'] = state['logger']
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
        # learning_alg picks the SquareCB online oracle:
        #   lin/rff/nonlinear -> torch regressor (online SGD on a sigmoid head)
        #   gb5/gb2           -> online gradient-boosted tree oracle (sklearn)
        #   xgb/xgb5/xgb2     -> online XGBoost oracle (incremental booster)
        # 'tree' has no online analogue here and falls back to 'linear'.
        _MODEL_TYPE = {'lin': 'linear', 'rff': 'rff', 'nonlinear': 'nonlinear',
                       'gb5': 'gb5', 'gb2': 'gb2',
                       'xgb': 'xgb', 'xgb5': 'xgb5', 'xgb2': 'xgb2'}
        sqcb_model_type = _MODEL_TYPE.get(Args.learning_alg, 'linear')
        # Append a CamelCased sampler suffix when not the default so that
        # different samplers don't overwrite each other's saved files
        # (and stay compatible with the notebook's [a-zA-Z][a-zA-Z0-9]*
        # learner regex). marginal: "lin"; bandit: "linBandit".
        sqcb_file_alg = Args.learning_alg + (
            '' if Args.sampler == 'marginal'
            else Args.sampler[0].upper() + Args.sampler[1:])
        if Args.param is not None:
            if os.path.isfile(outdir + "sqcb_%s_%0.3f_rewards_%d.out" % (sqcb_file_alg, Args.param, Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r, reg, val_tmp) = S.play(Args.T, verbose=True, validate=Bval,
                                       params=attach_logger({'gamma_multiplier': Args.param,
                                                             'model_type': sqcb_model_type,
                                                             'sampler': Args.sampler,
                                                             'y_scale': Args.y_scale,
                                                             'seed': Args.I,
                                                             'log_every': Args.log_every,
                                                             **_xgb_overrides(Args)}))
            stop = time.time()
            np.savetxt(outdir + "sqcb_%s_%0.3f_rewards_%d.out" % (sqcb_file_alg, Args.param, Args.I), r)
            np.savetxt(outdir + "sqcb_%s_%0.3f_validation_%d.out" % (sqcb_file_alg, Args.param, Args.I), val_tmp)
            np.savetxt(outdir + "sqcb_%s_%0.3f_time_%d.out" % (sqcb_file_alg, Args.param, Args.I), np.array([stop - start]))
        else:
            if os.path.isfile(outdir + "sqcb_%s_default_rewards_%d.out" % (sqcb_file_alg, Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r, reg, val_tmp) = S.play(Args.T, verbose=True, validate=Bval,
                                       params=attach_logger({'gamma_multiplier': 1.0,
                                                             'model_type': sqcb_model_type,
                                                             'sampler': Args.sampler,
                                                             'y_scale': Args.y_scale,
                                                             'seed': Args.I,
                                                             'log_every': Args.log_every,
                                                             **_xgb_overrides(Args)}))
            stop = time.time()
            np.savetxt(outdir + "sqcb_%s_default_rewards_%d.out" % (sqcb_file_alg, Args.I), r)
            np.savetxt(outdir + "sqcb_%s_default_validation_%d.out" % (sqcb_file_alg, Args.I), val_tmp)
            np.savetxt(outdir + "sqcb_%s_default_time_%d.out" % (sqcb_file_alg, Args.I), np.array([stop - start]))

    if Args.alg == "regcb":
        # RegCB is a contextual-bandit (L=1) algorithm; reject larger L early.
        if Args.L != 1:
            print("RegCB requires --L 1 (contextual bandit, not semibandit).")
            sys.exit(1)
        if Args.learning_alg not in ("lin", "gb5"):
            print("RegCB v1 supports --learning_alg in {'lin', 'gb5'}; "
                  f"got {Args.learning_alg!r}")
            sys.exit(1)
        beta = 0.1 if Args.param is None else float(Args.param)
        # Filename keeps mode + oracle + beta to disambiguate runs.
        regcb_tag = f"{Args.regcb_mode}_{Args.learning_alg}"
        out_prefix = outdir + "regcb_%s_%0.3f" % (regcb_tag, beta)
        if os.path.isfile(out_prefix + "_rewards_%d.out" % Args.I):
            print('---- ALREADY DONE ----')
            sys.exit(0)
        R = RegCB(B, learning_alg=Args.learning_alg, classification=False)
        params = attach_logger({
            'param': beta,
            'beta': beta,
            'regcb_mode': Args.regcb_mode,
            'learning_alg_name': Args.learning_alg,
            'regcb_warmup': Args.regcb_warmup,
            'regcb_binsearch_steps': Args.regcb_binsearch_steps,
            'regcb_synth_weight': Args.regcb_synth_weight,
            'regcb_reward_min': Args.regcb_reward_min,
            'regcb_reward_max': Args.regcb_reward_max,
            'regcb_ridge_alpha': Args.regcb_ridge_alpha,
            'regcb_gb_n_estimators': Args.regcb_gb_n_estimators,
            'regcb_refit_per_step': Args.regcb_refit_per_step,
            'seed': Args.I,
            'log_every': Args.log_every,
        })
        start = time.time()
        (r, reg, val_tmp) = R.play(Args.T, verbose=True, validate=Bval, params=params)
        stop = time.time()
        np.savetxt(out_prefix + "_rewards_%d.out" % Args.I, r)
        np.savetxt(out_prefix + "_validation_%d.out" % Args.I, val_tmp)
        np.savetxt(out_prefix + "_time_%d.out" % Args.I, np.array([stop - start]))

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

    if state['logger'] is not None:
        state['logger'].finish()

    print("---- DONE ----")


if __name__ == '__main__':
    main()
