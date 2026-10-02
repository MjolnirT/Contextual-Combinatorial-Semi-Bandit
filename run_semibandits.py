"""
Implementation backing the root-level ``run_semibandits.py`` command.

For new runs, invoke the root-level command, e.g.:
    python run_semibandits.py --T 30000 --dataset mslr30k --L 3 --I 0 --alg mini --learning_alg gb2 --param 0.01
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
_PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Simulators
import settings

from algorithms import (EpsGreedy, EpsGreedyValue, LinUCB, MiniMonster,
                        SquareCB, SquareCBLin, RegCB, RandomPolicy,
                        SkylinePolicy)


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
                        choices=['mini', 'eps', 'epsval', 'lin', 'sqcb', 'regcb',
                                 'random', 'skyline'])
    parser.add_argument('--learning_alg', action='store', default=None,
                        choices=[None, 'gb2', 'gb5', 'dt2', 'dt5',
                                 'tree', 'lin', 'rff', 'nonlinear',
                                 'xgb2', 'xgb5'])
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
                        choices=['marginal', 'depround', 'logdet'],
                        help="SquareCB sampler: 'marginal' (per-arm IGW "
                             "marginals + Madow systematic, default), "
                             "'depround' (per-arm IGW marginals + dependent "
                             "rounding), or "
                             "'logdet' (simplified SquareCB.Lin: "
                             "convex log-det-barrier program over the "
                             "C(K,L) super-action simplex, Frank-Wolfe "
                             "solver, direct categorical draw).")
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
    # Refit ablation (gb2/gb5/xgb2/xgb5 oracles). Tri-state: omitted keeps
    # the pre-ablation behavior AND the pre-ablation output filenames, so
    # the original campaign stays reproducible and collision-free.
    #   true  -> rebuild the model from scratch on the LATEST EPISODE only
    #            (rows since the previous training point)
    #   false -> extend the existing model (gb: sklearn warm_start, +100
    #            trees per point; xgb: append trees to the running booster)
    # Output files gain a 'Refit'/'Norefit' suffix on the oracle token.
    parser.add_argument('--refit', choices=['true', 'false'], default=None,
                        help="Refit ablation for the gb/xgb oracles; omit "
                             "for the pre-ablation behavior.")
    parser.add_argument('--refit_num_boost_round', type=int, default=None,
                        help="Trees per fresh booster when --refit true "
                             "(xgb oracles only; oracle default 100).")
    parser.add_argument('--results_root', default='results',
                        help="Output tree root, relative to the repo root. "
                             "Use a separate root (e.g. "
                             "results_refit_ablation) to keep an ablation's "
                             "traces out of the main campaign.")
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
    parser.add_argument('--regcb_cheat', dest='regcb_cheat',
                        action='store_true', default=True,
                        help="(Migrated RegCB) Use closed-form pred_range for "
                             "confidence intervals. Default: True.")
    parser.add_argument('--no-regcb_cheat', dest='regcb_cheat',
                        action='store_false',
                        help="(Migrated RegCB) Disable closed-form; use the "
                             "paper-faithful binary search. --param then "
                             "controls the radius beta. Lin oracle only "
                             "(gb5's fit_incremental is unimplemented).")
    parser.add_argument('--regcb_prec', type=float, default=0.001,
                        help="(Migrated RegCB) Confidence width / binary "
                             "search precision. Used as delta in pred_range "
                             "when cheat=True. Default: 0.001 (matches the "
                             "RegressorUCB prototype).")
    return parser.parse_args()


def _xgb_overrides(Args):
    """Collect XGBoost-oracle kwargs explicitly set on the CLI."""
    keys = ('max_depth', 'learning_rate', 'min_child_weight',
            'fit_every', 'warmup', 'num_boost_round',
            'buffer_size', 'reg_lambda', 'reg_alpha')
    return {k: v for k in keys
            if (v := getattr(Args, 'xgb_' + k)) is not None}


def _refit_overrides(Args):
    """Refit-ablation kwargs, empty when --refit is omitted.

    Returning {} in the omitted case matters: ``make_oracle`` then never
    passes ``refit``, the oracles keep their ``refit=None`` legacy path,
    and the pre-ablation traces stay bit-reproducible.
    """
    if getattr(Args, 'refit', None) is None:
        return {}
    out = {'refit': Args.refit == 'true'}
    if getattr(Args, 'refit_num_boost_round', None) is not None:
        out['refit_num_boost_round'] = Args.refit_num_boost_round
    return out


def _refit_suffix(Args):
    """CamelCase filename tag for the refit arm ('' when not ablating)."""
    if getattr(Args, 'refit', None) is None:
        return ''
    return 'Refit' if Args.refit == 'true' else 'Norefit'


def _fmt(p):
    """Lossless, collision-free param formatter for output filenames.

    Uses Python's shortest unique float repr (str(float)), which guarantees
    two distinct floats never produce the same string — unlike fixed-precision
    formats (%0.3f used to collapse mu=0.0001 and mu=0.0003 to '0.000').
    """
    return str(float(p))


def _run_and_save(out_prefix, seed, play_fn):
    """Short-circuit if `<out_prefix>_rewards_<seed>.out` exists, otherwise
    invoke `play_fn()` and write its 3 outputs.

    `play_fn` is a no-arg callable that returns `(r, reg, val_tmp)`
    matching `Semibandit.play()`'s contract. Encapsulates the save
    boilerplate that was duplicated across mini/eps/sqcb/lin/regcb/random.
    """
    rewards_path = f"{out_prefix}_rewards_{seed}.out"
    if os.path.isfile(rewards_path):
        print('---- ALREADY DONE ----')
        sys.exit(0)
    start = time.time()
    (r, reg, val_tmp) = play_fn()
    stop = time.time()
    np.savetxt(rewards_path, r)
    np.savetxt(f"{out_prefix}_validation_{seed}.out", val_tmp)
    np.savetxt(f"{out_prefix}_time_{seed}.out", np.array([stop - start]))


def make_learning_alg(name):
    if name == "gb2":
        return lambda: sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=2)
    if name == "gb5":
        return lambda: sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=5)
    if name == "tree":
        return lambda: sklearn.tree.DecisionTreeRegressor(max_depth=2)
    if name == "lin":
        return lambda: sklearn.linear_model.LinearRegression()
    if name in ("rff", "nonlinear", "xgb5", "xgb2", "dt", "dt5", "dt2"):
        # SquareCB-only model-type selectors; legacy learning_alg gate
        # in the sqcb branch needs a non-None marker, but the actual
        # regressor factory is not used (online oracle takes over).
        return name
    return None


def main():
    Args = parse_args()
    print(Args, flush=True)
    _root = Args.results_root
    if Args.noise is not None:
        Args.noise = float(Args.noise)
        outdir = os.path.join(_PROJECT_ROOT, '%s/%s_T=%d_L=%d_e=%0.1f/' % (_root, Args.dataset, Args.T, Args.L, Args.noise))
    else:
        outdir = os.path.join(_PROJECT_ROOT, '%s/%s_T=%d_L=%d_e=0.0/' % (_root, Args.dataset, Args.T, Args.L))
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    # random / skyline carry no tunable param; the SLURM templates still pass
    # --param 'na' for them (the TSV always has a param column). Leave it
    # unparsed in that case so float('na') doesn't blow up.
    if Args.param is not None and str(Args.param).lower() != 'na':
        Args.param = float(Args.param)
    elif str(Args.param).lower() == 'na':
        Args.param = None

    B = Simulators.DatasetBandit(dataset=Args.dataset, L=Args.L, loop=True, metric=None, noise=Args.noise)

    # Bval (held-out validation) is constructed only if the vali file is on
    # disk. Under the Arxiv protocol we ship only the whole train file and
    # skip held-out evaluation; under the legacy NeurIPS protocol the vali
    # file is present and Bval is built as before.
    try:
        Bval = Simulators.DatasetBandit(dataset=Args.dataset, L=Args.L,
                                        loop=False, metric=None,
                                        noise=Args.noise, train=False)
    except (FileNotFoundError, OSError) as e:
        print(f"No validation set on disk ({e}); skipping Bval.", flush=True)
        Bval = None

    # Query-order shuffling. If the per-seed permutation file
    # <dataset>_train_<I>.npz is on disk (legacy NeurIPS layout), load it;
    # otherwise generate the permutation on the fly from a seed-derived
    # RNG (Arxiv layout — only the whole train file is shipped).
    _PERM_FILE = {
        'mslr30k': settings.DATA_DIR + "mslr/mslr30k_train_%d.npz" % (Args.I),
        'yahoo':   settings.DATA_DIR + "yahoo/yahoo_train_%d.npz" % (Args.I),
    }.get(Args.dataset)
    if Args.dataset in ('mslr30k', 'yahoo') and Args.I < 20 and _PERM_FILE \
            and os.path.exists(_PERM_FILE):
        order = np.load(_PERM_FILE)
        print("Setting order for Iteration %d from %s" % (Args.I, _PERM_FILE), flush=True)
        o = order['order']
        B.contexts.order = o[np.where(o < B.contexts.X)[0]]
        B.contexts.curr_idx = 0
    elif Args.dataset in ('mslr30k', 'yahoo'):
        rng = np.random.default_rng(Args.I)
        print("Generating on-the-fly order for Iteration %d (X=%d)"
              % (Args.I, B.contexts.X), flush=True)
        B.contexts.order = rng.permutation(B.contexts.X)
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
        _suffix = _fmt(Args.param) if Args.param is not None else "default"
        _params = {'delta': Args.param} if Args.param is not None else {}
        _run_and_save(
            outdir + f"lin_{_suffix}", Args.I,
            lambda: L.play(Args.T, verbose=True, validate=Bval,
                           params=attach_logger(_params)),
        )

    if Args.alg == "mini":
        if learning_alg is None:
            print("Cannot run MiniMonster without learning algorithm")
            sys.exit(1)
        M = MiniMonster(B, learning_alg=learning_alg, classification=False,
                        lin_cache=(Args.learning_alg == "lin"))
        _suffix = _fmt(Args.param) if Args.param is not None else "default"
        _params = {'weight': np.ones(Args.L)}
        if Args.param is not None:
            _params['mu'] = Args.param
        _run_and_save(
            outdir + f"mini_{Args.learning_alg}_{_suffix}", Args.I,
            lambda: M.play(Args.T, verbose=True, validate=Bval,
                           params=attach_logger(_params)),
        )

    if Args.alg == "eps":
        if learning_alg is None:
            print("Cannot run EpsGreedy without learning algorithm")
            sys.exit(1)
        E = EpsGreedy(B, learning_alg=learning_alg, classification=False)
        _suffix = _fmt(Args.param) if Args.param is not None else "default"
        _params = {'weight': np.ones(Args.L), 'train_all': True}
        if Args.param is not None:
            _params['eps'] = Args.param
        _run_and_save(
            outdir + f"epsall_{Args.learning_alg}_{_suffix}", Args.I,
            lambda: E.play(Args.T, verbose=True, validate=Bval,
                           params=attach_logger(_params)),
        )

    if Args.alg == "epsval":
        if learning_alg is None:
            print("Cannot run EpsGreedyValue without learning algorithm")
            sys.exit(1)
        EV = EpsGreedyValue(B, learning_alg=learning_alg, classification=False)
        _MODEL_TYPE = {'lin': 'linear', 'rff': 'rff', 'nonlinear': 'nonlinear',
                       'gb5': 'gb5', 'gb2': 'gb2',
                       'dt5': 'dt5', 'dt2': 'dt2',
                       'xgb5': 'xgb5', 'xgb2': 'xgb2'}
        epsval_model_type = _MODEL_TYPE.get(Args.learning_alg, 'linear')
        _suffix = _fmt(Args.param) if Args.param is not None else "default"
        _eps = Args.param if Args.param is not None else 0.1
        _params = {'eps': _eps,
                   'model_type': epsval_model_type,
                   'y_scale': Args.y_scale,
                   'seed': Args.I,
                   'log_every': Args.log_every,
                   **_xgb_overrides(Args)}
        _run_and_save(
            outdir + f"epsval_{Args.learning_alg}_{_suffix}", Args.I,
            lambda: EV.play(Args.T, verbose=True, validate=Bval,
                            params=attach_logger(_params)),
        )

    if Args.alg == "sqcb":
        if learning_alg is None:
            print("Cannot run SquareCB without learning algorithm")
            sys.exit(1)
        # sampler=logdet is SquareCB.lin (FGMZ 2021): its own class with
        # BANDIT feedback — the oracle sees one (summed features, slate
        # reward) sample per round instead of per-arm rewards. All other
        # samplers keep semi-bandit SquareCB.
        _sqcb_cls = SquareCBLin if Args.sampler == 'logdet' else SquareCB
        S = _sqcb_cls(B, learning_alg=learning_alg, classification=False)
        # learning_alg picks the SquareCB online oracle:
        #   lin/rff/nonlinear -> torch regressor (online SGD on a sigmoid head)
        #   gb5/gb2           -> sklearn GradientBoostingRegressor refit on
        #                        RegCB's burn_in + 2^k schedule (same backbone
        #                        as RegCB's gb5)
        #   dt5/dt2           -> hand-rolled online decision-tree residual
        #                        ensemble (the previous gb5 implementation)
        #   xgb5/xgb2         -> online XGBoost oracle (incremental booster,
        #                        depth 5 / 2). The bare 'xgb' alias was
        #                        dropped — it silently meant depth=5 and so
        #                        duplicated 'xgb5'.
        # 'tree' has no online analogue here and falls back to 'linear'.
        _MODEL_TYPE = {'lin': 'linear', 'rff': 'rff', 'nonlinear': 'nonlinear',
                       'gb5': 'gb5', 'gb2': 'gb2',
                       'dt5': 'dt5', 'dt2': 'dt2',
                       'xgb5': 'xgb5', 'xgb2': 'xgb2'}
        sqcb_model_type = _MODEL_TYPE.get(Args.learning_alg, 'linear')
        # Append a CamelCased sampler suffix when not the default so that
        # different samplers don't overwrite each other's saved files
        # (and stay compatible with the notebook's [a-zA-Z][a-zA-Z0-9]*
        # learner regex). marginal: "lin"; depround: "linDepround".
        # Refit tag sits between the oracle and the sampler suffix, so a
        # token parses as <oracle><Refit|Norefit><SamplerSuffix> and both
        # tags can be stripped independently.
        sqcb_file_alg = Args.learning_alg + _refit_suffix(Args) + (
            '' if Args.sampler == 'marginal'
            else Args.sampler[0].upper() + Args.sampler[1:])
        _suffix = _fmt(Args.param) if Args.param is not None else "default"
        _gamma = Args.param if Args.param is not None else 1.0
        _params = {'gamma_multiplier': _gamma,
                   'model_type': sqcb_model_type,
                   'sampler': Args.sampler,
                   'y_scale': Args.y_scale,
                   'seed': Args.I,
                   'log_every': Args.log_every,
                   **_xgb_overrides(Args),
                   **_refit_overrides(Args)}
        _run_and_save(
            outdir + f"sqcb_{sqcb_file_alg}_{_suffix}", Args.I,
            lambda: S.play(Args.T, verbose=True, validate=Bval,
                           params=attach_logger(_params)),
        )

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
        regcb_tag = f"{Args.regcb_mode}_{Args.learning_alg}"
        R = RegCB(B, learning_alg=Args.learning_alg, classification=False)
        _params = attach_logger({
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
            'regcb_cheat': Args.regcb_cheat,
            'regcb_prec': Args.regcb_prec,
            'seed': Args.I,
            'log_every': Args.log_every,
        })
        _run_and_save(
            outdir + f"regcb_{regcb_tag}_{_fmt(beta)}", Args.I,
            lambda: R.play(Args.T, verbose=True, validate=Bval, params=_params),
        )

    if Args.alg == "random":
        # Uniform-random slate baseline; no learner, no knob. Seeded per-I so
        # the reward trace is reproducible for a seed and differs across seeds.
        R = RandomPolicy(B)
        _run_and_save(
            outdir + "random", Args.I,
            lambda: R.play(Args.T, verbose=True, validate=Bval,
                           params=attach_logger({'seed': Args.I})),
        )

    if Args.alg == "skyline":
        # Supervised-learning skyline: fit the oracle once on the WHOLE train
        # set, then greedily pick top-L in-sample. Upper-bound reference; no
        # tunable param. Output file: skyline_<oracle>_rewards_<I>.out.
        if Args.learning_alg is None:
            print("Cannot run skyline without learning algorithm")
            sys.exit(1)
        SK = SkylinePolicy(B, learning_alg=Args.learning_alg)
        _run_and_save(
            outdir + f"skyline_{Args.learning_alg}", Args.I,
            lambda: SK.play(Args.T, verbose=True, validate=Bval,
                            params=attach_logger({})),
        )

    if state['logger'] is not None:
        state['logger'].finish()

    print("---- DONE ----")


if __name__ == '__main__':
    main()
