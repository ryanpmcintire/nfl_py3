# Projected lineups on This Week

The This Week game deep dive includes an optional **Projected lineups & model
impact** panel. It is a static view designed for GitHub Pages: a refresh job
builds `artifacts/lineups/current/lineups.json`, and the renderer only reads
that artifact. The artifact is ignored by Git along with the other local data
and model outputs. The lineup file is REPLACED on every refresh, never
accumulated: the builder overwrites the stable path atomically and removes
superseded stamped `*/lineups.json` runs. History lives in the depth-chart
snapshot the payload cites (`depth_snapshot`), not in display copies.

The scheduled refresh runs Monday through Sunday at noon Eastern. It snapshots
the current depth chart, rebuilds the player features and weekly forecast from
that snapshot, then creates the final lineup artifact linked to the new
forecast. Run the same flow locally with:

```powershell
.\.tools\uv.exe run --no-sync python scripts\refresh_lineup_forecast.py
```

The panel shows the complete latest depth-chart snapshot — offense, defense,
and special teams in their own sections with per-section player counts —
along with source timestamps, injury-feed availability, and the active model's
scored QB family contribution when the waterfall contains it. Unit filter
buttons above the panel toggle each section's visibility for the current
reader only; every player row stays in the published HTML. The renderer
refuses to publish the panel beside a forecast whose projected QB is absent
from that lineup snapshot.

Play probability is only shown when it is present in the model artifact. A
missing injury feed or probability is displayed as unavailable rather than
estimated. Player rows outside the scored model family are labeled context-only
so readers can distinguish lineup context from a model input.

## UI-20 (2026-09-05): every listed player, not only the QB

Before this change, `play_probability` was only ever set for the one QB the
active model consumed (`model_role: "base_model"`); every other listed
starter had `model_role: "context_only"` and no probability at all -- the
owner's complaint that prompted this section.

## UI-20-AB (2026-09-05): a real forecast, not a base rate

The first fix (above) gave every player a number, but it came from a
position-level **no-designation base rate** for anyone without a visible
injury designation -- which produced its own owner complaint: a rookie
backup QB with no designation read 47%, a healthy veteran backup read 95%,
because both were the SAME constant, keyed only on position group and
recent role, not on the two players themselves. A same-day interim render
rule then hid the percentage entirely for undesignated players (showing
"no designation" instead) as a stopgap while a real model was built.

That stopgap is now retired. Every player with a `gsis_id` carries a
`play_probability` (and, for QBs, a `start_probability`) from
`nfl_ats.play_probability` -- a walk-forward, isotonic-calibrated
gradient-boosting model of P(plays) and P(starts) trained on depth-chart
rank, this week's own injury report, recent playing-time history, roster
status, and (for backup QBs) the team's own QB1's injury status. See
`docs/play_probability_model.md` for the model itself (features, walk-forward
protocol, measured Brier improvement, calibration table). `probability_source`
says exactly how each row's number was produced:

- **`play_probability_model`** -- every player with a `gsis_id`, designated
  or not: a real per-player, per-game forecast that considers depth chart,
  this week's injury report, and recent snaps -- the owner's own directive
  ("it needs to be a forecast about the game and it needs to consider depth
  chart"). `model_qb_start_probability` separately preserves the active
  margin model's own `{side}_qb_start_probability` input for the one QB it
  actually consumed (previously `play_probability` itself for that player
  under the retired `base_model_qb` source) -- kept as its own field, not
  deleted.
- **`unavailable`** -- reserved for a depth-chart row this machinery
  genuinely cannot score: no `gsis_id`, or no predictor was available this
  run. `play_probability` stays `None` and the panel still shows an em dash
  -- never an invented number.

Every player also carries a short `probability_reason` naming exactly which
rule fired; the rendered panel exposes it as a tooltip on the percentage, and
prints a one-line legend once per lineup block. **Every player with a model
probability now shows it as a percentage** -- the panel no longer hides a
number for lack of an injury designation; when a designation exists this
week it is shown next to the player's name instead, alongside their injury
status. The QB slot alone also shows a second, smaller "start" number
(`start_probability`: the same model's chance this player is the one who
starts, distinct from whether he takes the field at all). Percentages stay
coloured by availability risk only (green >=85%, amber 50-85%, red <50%);
the em dash is reserved for `probability_source == "unavailable"`. The
lineup-aware assistant (`nfl_ats.board_assistant_lineups.player_availability_answer`)
answers an "is `<player>` playing" question for any listed player, names the
source in its answer text, and quotes the same real percentage regardless of
designation.

**Known limitation, measured, not this feature's fault:** the active
model's own `{side}_qb_start_probability` (i.e. `model_qb_start_probability`
in the artifact) is currently a dead constant `1.0` for every 2026 game
(verified against the live `artifacts/lineups/current/lineups.json` and the
weekly forecast) because the production feature table has no 2025/2026
injury coverage at all -- the pre-existing ENG-39 defect
(`docs/injury_timestamp_fallback.md`). Fixing that is a separate,
owner-scheduled production rebuild; the `play_probability_model` numbers for
every player (QB included) are computed fresh at build time and are
unaffected -- they show real variation across the full roster.
