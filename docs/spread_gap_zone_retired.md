# Spread-gap zone retired from the played card ? 2026-09-07

**Read ? owner directive in AGENTS.md, ?Football margins are multimodal?:**
?no more arbitrary pick flips because we can't explain a drop in accuracy at
certain point thresholds... we need to be able to explain these things so we
can understand the weak points in the model?.

**Read ? `src/nfl_ats/four_overlay_composition.py`:** the played OR union now
contains coach fade, division revenge and player arrests. These adjustments
name mechanisms. The spread-gap zone's mined 7.5?10 threshold alone does not
explain a mechanism, so it no longer flips the played card. The new identity
is `overlay_union_coach_division_revenge_player_arrests_v2`; the fingerprint
is derived from its three-member definition. Shared function names retain
?four_overlay? for caller compatibility.

**Read ? `artifacts/overlay_subset_composition/20260907T152039959302Z/result.json`,
`subsets` rows matched by member set:** on 1,503 scored opener games, the
retired four-member union scores 55.4225%, the three-member union 55.2229%,
and the model baseline 53.9587%. The cost is 0.1996 accuracy points on the
same mined, selected archive. It is not independent evidence or a reason to
retain a mechanism-free flip. The three-member-versus-model season-blocked
interval is [-0.1916, +2.5049] accuracy points, `probability_positive=0.9511`;
the corresponding four-member interval is [-1.5464, +3.8332],
`probability_positive=0.83875`. These are separate comparisons to the model,
not uncertainty for the retirement difference.

**Read ? `artifacts/prospective/challengers.json` and
`src/nfl_ats/retired_four_member_union.py`:** the existing standalone
`spread_gap_zone_fade_overlay` remains dual-tracked without altering its
ledger. The added `overlay_four_member_union_retired_20260907` challenger
reconstructs the former card from the primary paper ledger's frozen model
side, decision spread and three-member flip flag. It unions the zone flag
and complements once, so overlapping rules never cancel. Publication records
both arms with the established fail-open challenger pattern and first-write-
wins semantics. The earlier coach-to-arrests incumbent keeps its own identity.

**Read ? `docs/spread_regime_program.md` (lane H's work):** MOD-18 studies
spread-regime calibration and the discrete key-number margin distribution
as the replacement modeling program. **Inferred:** explaining and fixing
those weaknesses is preferable to hiding them behind threshold reversals.
This retirement is an owner-directed playing-policy change, not a terminal
statistical verdict on the tracked signal. No research line is closed.
