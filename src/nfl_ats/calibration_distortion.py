"""Primitives for measuring what the residual/calibration step does to a measured effect.

Research item: ``docs/calibration_distortion.md``. Defect **D1** in
``docs/revisit_list.md`` claimed the residual/calibration step can attenuate or
even invert a small effect: a planted +1.3 accuracy points came back -0.7 and
a planted +3.0 came back +1.14 when scored through the full pipeline, while a
"sign-only" ablation recovered both closely
(``docs/purged_cv.md``, positive-control section). **This module's own
experiment disproved that claim.** ``scripts/purged_validate.py`` called
``inject_synthetic_signal(..., seed=42)`` once per magnitude, and that
function derives BOTH the signal and the noise from ``seed`` alone
(``purged_cv.py:651-655``), so the two recorded rows are one plant draw
reported twice, not two independent findings. Replicated across 21 seeds the
effect is about -0.2 points, bounded under ~0.35 against the claimed 2.0-point
swing; on the realistic additive construction (both arms carry the real
target, as every actual verdict does) the full pipeline recovers **0.964
[0.893, 1.036]** of the achievable delta. D1 is not real at the magnitude it
was written from (``docs/calibration_distortion.md`` sec 8). This module
supplies the exact, cheap primitives that experiment needs, so the driver
script never re-derives the production rule by hand.

What the production forced pick actually is
-------------------------------------------
``MarginModel.predict`` builds ``center + residuals`` for every game and reads
``home_cover_probability = margin._smoothed_probability(center + residuals,
line) = (count(center + residuals > line) + 0.5) / (len + 1)``. The forced pick
is ``home_cover_probability >= 0.5``. Substituting ``predicted_market_residual
= center - line`` and rearranging, that inequality is exactly

    count(residuals > -predicted_market_residual) >= n / 2

which is a pure THRESHOLD on ``predicted_market_residual`` at
``-residuals_sorted[n - ceil(n / 2)]`` -- one order statistic of the residual
sample, near its median. So the full pipeline's pick and the sign-only
ablation's pick differ by exactly one number per fitted model: a LOCATION
offset. :func:`implied_pick_threshold` returns it, and
:func:`full_pipeline_home_pick` / :func:`sign_only_home_pick` return the two
picks. ``tests/test_calibration_distortion.py`` pins both against a real
``MarginModel.predict`` call rather than against this docstring's algebra.

Why the planting helper is here
-------------------------------
``purged_cv.inject_synthetic_signal`` OVERWRITES ``ats_margin`` with
``beta * signal + noise``, so the planted column becomes the target's only
informative ingredient and every real relationship in the data is destroyed.
That is the right construction for asking "can the CV scheme size a known
effect", and the wrong one for asking "does the calibration step distort the
PAIRED DELTAS this project's verdicts are actually made of", because the
baseline arm of such a comparison is a pure null and the residual sample has
no real target drift left to correct. :func:`plant_additive_effect` is the
complement: it ADDS a known effect to the real ``ats_margin`` instead of
replacing it, so both arms of the paired comparison still carry the real
market signal and the calibration sample still sees real drift.

Scope: CFB-only empirical work (rotation-registry rule 8, no NFL window).
Nothing here is imported by ``margin.py``, ``cfb_benchmark.py``,
``experiments.py``, or the active model; importing this module cannot change
a published probability or a pick.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.data import require_columns
from nfl_ats.purged_cv import synthetic_signal_beta

__all__ = [
    "additive_plant_gamma",
    "full_pipeline_home_pick",
    "implied_pick_threshold",
    "orthogonal_carrier",
    "plant_additive_effect",
    "sign_only_home_pick",
    "standardized_carrier",
]


def implied_pick_threshold(residuals: npt.ArrayLike) -> float:
    """The ``predicted_market_residual`` value at which the production pick flips.

    ``full_pipeline_home_pick(residuals, pr)`` is exactly
    ``pr > implied_pick_threshold(residuals)`` (pinned by
    ``test_implied_threshold_reproduces_full_pipeline_pick``). The value is
    ``-residuals_sorted[n - ceil(n / 2)]``, i.e. minus a near-median order
    statistic of the residual sample -- so a residual sample whose location
    sits at ``+m`` moves the pick threshold to ``-m``, favouring the home side
    on every game whose predicted residual lands in ``(-m, 0]``.
    """

    values = np.sort(np.asarray(residuals, dtype=np.float64))
    n = int(values.size)
    if n == 0:
        raise ValueError("implied_pick_threshold requires at least one residual")
    return float(-values[n - math.ceil(n / 2)])


def full_pipeline_home_pick(
    residuals: npt.ArrayLike, predicted_market_residual: npt.ArrayLike
) -> npt.NDArray[np.bool_]:
    """The production forced pick: ``home_cover_probability >= 0.5``, vectorised.

    Computed from ``margin._smoothed_probability``'s own inequality rather
    than by calling ``MarginModel.predict``, which loops per row in Python and
    additionally computes four quantiles and a three-way push split this
    experiment never reads. Equality with ``MarginModel.predict`` is a test,
    not an assumption.
    """

    sorted_residuals = np.sort(np.asarray(residuals, dtype=np.float64))
    n = int(sorted_residuals.size)
    if n == 0:
        raise ValueError("full_pipeline_home_pick requires at least one residual")
    thresholds = -np.asarray(predicted_market_residual, dtype=np.float64)
    count_above = n - np.searchsorted(sorted_residuals, thresholds, side="right")
    return np.asarray(count_above >= n / 2.0, dtype=bool)


def sign_only_home_pick(predicted_market_residual: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """The sign-only ablation: ``sign(predicted_margin - spread_line) > 0``.

    Identical training set, identical ridge fit, identical predicted centre --
    it only skips reading the out-of-time residual sample's location.
    """

    return np.asarray(np.asarray(predicted_market_residual, dtype=np.float64) > 0.0, dtype=bool)


def orthogonal_carrier(n: int, *, seed: int) -> npt.NDArray[np.float64]:
    """An iid ``{-1, +1}`` carrier column, independent of every real covariate.

    The same draw ``purged_cv.inject_synthetic_signal`` uses, exposed on its
    own so an additive plant can reuse the identical carrier distribution and
    the orthogonal-vs-collinear contrast differs in one thing only.
    """

    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    return np.asarray(rng.choice(np.array([-1.0, 1.0]), size=n), dtype=np.float64)


def standardized_carrier(frame: pd.DataFrame, column: str) -> npt.NDArray[np.float64]:
    """A REAL feature column, median-imputed and scaled to mean 0 / sd 1.

    Standardising is what makes a real carrier's planted ``gamma``
    commensurable with :func:`orthogonal_carrier`'s ``{-1, +1}`` draw (both
    have unit scale), so the only difference between the two constructions is
    the carrier's correlation with the rest of the design.
    """

    if column not in frame.columns:
        raise KeyError(f"Carrier column {column!r} is not in the frame")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"Carrier column {column!r} has no finite values")
    values = np.where(np.isfinite(values), values, float(np.median(finite)))
    scale = float(np.std(values))
    if scale <= 0.0:
        raise ValueError(f"Carrier column {column!r} is constant")
    return np.asarray((values - float(np.mean(values))) / scale, dtype=np.float64)


def additive_plant_gamma(target_accuracy: float, noise_std: float) -> float:
    """``gamma`` for an additive plant, on ``purged_cv``'s own accuracy scale.

    Reuses ``synthetic_signal_beta``'s parameterisation --
    ``gamma = noise_std * Phi^-1(target_accuracy)`` -- so an additive plant at
    "1.3 points" and the existing overwrite plant at "1.3 points" carry the
    same coefficient on a unit-scale carrier and the two constructions differ
    only in whether the real target survives underneath.
    """

    return synthetic_signal_beta(target_accuracy, noise_std)


def plant_additive_effect(
    frame: pd.DataFrame,
    *,
    carrier: npt.ArrayLike,
    gamma: float,
    column: str | None = None,
) -> pd.DataFrame:
    """Add ``gamma * carrier`` to the REAL ``ats_margin`` and re-derive the outcomes.

    The complement of ``purged_cv.inject_synthetic_signal``: that function
    replaces the target, this one perturbs it. Everything the model can learn
    from the real data survives, so the baseline arm of a paired comparison is
    a real model with real accuracy rather than a coin flip, and the
    calibration sample still carries whatever real target drift it normally
    carries. ``column`` optionally writes the carrier into the frame under
    that name so it can be added to a candidate arm's feature contract.
    """

    require_columns(frame, ("spread_line", "ats_margin"), "additive plant frame")
    values = np.asarray(carrier, dtype=np.float64)
    if values.size != len(frame):
        raise ValueError("carrier length must match the frame")
    if not np.isfinite(values).all():
        raise ValueError("carrier must be finite")
    if not math.isfinite(gamma):
        raise ValueError("gamma must be finite")

    working = frame.copy()
    base = pd.to_numeric(working["ats_margin"], errors="raise").to_numpy(dtype=float)
    planted = base + gamma * values
    spread = pd.to_numeric(working["spread_line"], errors="raise").to_numpy(dtype=float)
    working["ats_margin"] = planted
    working["result"] = spread + planted
    working["home_cover"] = np.select([planted > 0, planted < 0], [1.0, 0.0], default=np.nan)
    if column is not None:
        working[column] = values
    return working
