import numpy as np
import sim04_loso as base

THETA_BOUND = 2.0
BISECTION_STEPS = 80


def balance_weights(margins, line):
    diff = margins - line[:, None]
    cov = diff > base.ATOM_TOLERANCE
    los = diff < -base.ATOM_TOLERANCE
    lo = np.full(len(margins), -THETA_BOUND)
    hi = np.full(len(margins), THETA_BOUND)
    for _ in range(BISECTION_STEPS):
        theta = (lo + hi) / 2.0
        z = theta[:, None] * diff
        w = np.exp(z - z.max(1, keepdims=True))
        decided = (w * (cov | los)).sum(1)
        share = (w * cov).sum(1) / np.clip(decided, 1e-300, None)
        below = share < 0.5
        lo = np.where(below, theta, lo)
        hi = np.where(below, hi, theta)
    theta = (lo + hi) / 2.0
    z = theta[:, None] * diff
    w = np.exp(z - z.max(1, keepdims=True))
    w = w / w.sum(1, keepdims=True)
    ess = 1.0 / (w**2).sum(1)
    share = (w * cov).sum(1) / np.clip((w * (cov | los)).sum(1), 1e-300, None)
    return w * margins.shape[1], ess, share, theta
