"""
Online per-arm reward regressor for SquareCB / IGW semi-bandits.

Maps per-arm features phi_k in R^{ld_dim} to a reward score in [0,1] via a sigmoid head. Trained online with one BCE gradient step per round on the mini-batch of L played arms.
"""

import numpy as np
import torch
import torch.nn as nn

from math import pi, sqrt


def _make_lr_lambda(tzero):
    def lr_lambda(step):
        return 1.0 / np.sqrt(step / tzero + 1.0)
    return lr_lambda


class _LinearHead(nn.Module):
    def __init__(self, ld_dim):
        super().__init__()
        self.linear = nn.Linear(ld_dim, 1)

    def forward(self, phi):
        return self.linear(phi).squeeze(-1)


class _CauchyRFF(nn.Module):
    def __init__(self, ld_dim, numrff, sigma):
        super().__init__()
        W = torch.empty(ld_dim, numrff).cauchy_(sigma=sigma)
        b = 2 * pi * torch.rand(numrff)
        self.register_buffer("rffW", W)
        self.register_buffer("rffb", b)
        self.register_buffer("sqrtrff", torch.tensor(float(sqrt(numrff))))
        self.linear = nn.Linear(numrff, 1)

    def forward(self, phi):
        with torch.no_grad():
            rff = (phi @ self.rffW + self.rffb).cos() / self.sqrtrff
        return self.linear(rff).squeeze(-1)


class _ResidualBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(d, d))
        self.act = nn.LeakyReLU(0.01, inplace=False)

    def forward(self, x):
        return x + self.act(x @ self.W)


class _NonlinearHead(nn.Module):
    def __init__(self, ld_dim, depth):
        super().__init__()
        self.blocks = nn.Sequential(*[_ResidualBlock(ld_dim) for _ in range(depth)])
        self.linear = nn.Linear(ld_dim, 1)

    def forward(self, phi):
        return self.linear(self.blocks(phi)).squeeze(-1)


class OnlinePerArmRegressor:
    """Sigmoid-output per-arm regressor with one BCE step per call to ``step``."""

    def __init__(self, ld_dim, model_type='linear', initlr=0.05, tzero=100.0,
                 device='cpu', numrff=512, rff_sigma=1.0, depth=2, seed=None,
                 normalize_features=True, y_scale=1.0):
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(int(seed))

        if model_type == 'linear':
            net = _LinearHead(ld_dim)
        elif model_type == 'rff':
            net = _CauchyRFF(ld_dim, numrff, rff_sigma)
        elif model_type == 'nonlinear':
            net = _NonlinearHead(ld_dim, depth=depth)
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        self.model = net.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=initlr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt, _make_lr_lambda(tzero)
        )
        self.normalize_features = bool(normalize_features)
        self.y_scale = float(y_scale)
        if self.y_scale <= 0:
            raise ValueError(f"y_scale must be > 0, got {self.y_scale}")

    @staticmethod
    def _l2_normalize_rows(phi):
        # Per-row L2 normalization with a tiny floor so SGD operates on
        # unit-scale inputs regardless of dataset feature magnitudes.
        norms = np.linalg.norm(phi, axis=-1, keepdims=True)
        return phi / np.clip(norms, 1e-12, None)

    @torch.no_grad()
    def predict(self, features):
        f = np.asarray(features, dtype=np.float32)
        if self.normalize_features:
            f = self._l2_normalize_rows(f)
        phi = torch.as_tensor(f, dtype=torch.float32, device=self.device)
        scores = self.model(phi)
        # Predictions are returned in the original target scale: the
        # network learns y/y_scale internally; multiply back here so
        # downstream IGW sees scores comparable to true rewards.
        return scores.detach().cpu().numpy().astype(np.float64) * self.y_scale

    def step(self, phi_played, y_played):
        f = np.asarray(phi_played, dtype=np.float32)
        if self.normalize_features:
            f = self._l2_normalize_rows(f)
        phi = torch.as_tensor(f, dtype=torch.float32, device=self.device)
        # Target normalization: train on y/y_scale so the internal MSE
        # loss landscape stays in the [0,1]-ish regime regardless of the
        # raw reward magnitude.
        y = torch.as_tensor(np.asarray(y_played) / self.y_scale,
                            dtype=torch.float32, device=self.device)

        self.opt.zero_grad()
        scores = self.model(phi)
        loss = (scores - y).pow(2).mean()
        loss.backward()
        self.opt.step()
        self.scheduler.step()
        return float(loss.item())
