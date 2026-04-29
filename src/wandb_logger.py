"""
Wandb integration for semibandit experiments.

Defaults:
    entity  = "contextualbandit"
    project = "contextual-semi-bandit"

Usage:

    from wandb_logger import make_logger, init_run

    logger = init_run(args, params, tag="ablation-ips-off")

    alg.play(T, params={"logger": logger, ...})
    logger.finish()

If wandb is not installed or ``project`` is None, ``make_logger`` returns a
no-op logger and SquareCB silently runs without logging.
"""

from __future__ import annotations

import os
import warnings


WANDB_ENTITY = "contextualbandit"
WANDB_PROJECT = "contextual-semi-bandit"


class _NoopLogger:
    active = False

    def log(self, metrics, step=None):
        pass

    def finish(self):
        pass


class _WandbLogger:
    active = True

    def __init__(self, run):
        self.run = run

    def log(self, metrics, step=None):
        if step is not None:
            self.run.log(metrics, step=step)
        else:
            self.run.log(metrics)

    def finish(self):
        self.run.finish()


def make_logger(project=None, entity=None, name=None, config=None,
                mode=None, tags=None, group=None, dir=None, reinit=True):
    """Return a wandb logger, or a no-op when wandb is unavailable / disabled.

    Defaults to ``WANDB_ENTITY`` / ``WANDB_PROJECT`` constants when those
    args are left as None.
    """
    if mode == "disabled":
        return _NoopLogger()
    project = project or WANDB_PROJECT
    entity = entity or WANDB_ENTITY
    if project is None:
        return _NoopLogger()
    try:
        import wandb
    except ImportError:
        warnings.warn("wandb is not installed; falling back to no-op logger.",
                      RuntimeWarning, stacklevel=2)
        return _NoopLogger()

    run = wandb.init(
        entity=entity,
        project=project,
        name=name,
        config=config or {},
        tags=tags,
        group=group,
        dir=dir,
        mode=mode or os.environ.get("WANDB_MODE", "online"),
        reinit=reinit,
    )
    return _WandbLogger(run)


def init_run(args, params=None, tag=None):
    """Build a wandb run for any semibandit experiment from ``run_semibandits.py`` args.

    ``args`` is the parsed argparse Namespace (needs T, L, I, dataset, alg,
    learning_alg, param). ``params`` is the optional dict that will be
    forwarded to ``alg.play(T, params=...)``; its non-internal entries are
    captured in ``wandb.config`` under a ``params/`` prefix. Returns a logger
    that should be placed in ``params['logger']`` and ``finish()``-ed after
    the run.
    """
    params = params or {}
    parts = [args.dataset, f"T{args.T}", f"L{args.L}", f"I{args.I}", args.alg]
    if getattr(args, "learning_alg", None):
        parts.append(args.learning_alg)
    if getattr(args, "param", None) is not None:
        parts.append(f"p{args.param}")
    name = "-".join(parts)

    config = {
        "dataset": args.dataset,
        "T": args.T,
        "L": args.L,
        "I": args.I,
        "alg": args.alg,
        "learning_alg": getattr(args, "learning_alg", None),
        "param": getattr(args, "param", None),
    }
    for k, v in params.items():
        if k == "logger":
            continue
        config[f"params/{k}"] = v

    return make_logger(name=name, config=config,
                       tags=[tag] if tag else None)
