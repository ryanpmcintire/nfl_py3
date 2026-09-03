# Projected lineups on This Week

The This Week game deep dive includes an optional **Projected lineups & model
impact** panel. It is a static view designed for GitHub Pages: a refresh job
builds `artifacts/lineups/<stamp>/lineups.json`, and the renderer only reads
that artifact. The artifact is ignored by Git along with the other local data
and model outputs.

Build a local snapshot with:

```powershell
.\.tools\uv.exe run --no-sync python scripts\build_week_lineups.py
```

The panel shows current depth-chart starters, source timestamps, injury-feed
availability, and the active model's scored QB family contribution when the
waterfall contains it. A current depth-chart QB that differs from the forecast
input is intentionally rendered as a mismatch with “rerun forecast” guidance;
the display never silently changes a published pick or probability.

Play probability is only shown when it is present in the model artifact. A
missing injury feed or probability is displayed as unavailable rather than
estimated. Player rows outside the scored model family are labeled context-only
so readers can distinguish lineup context from a model input.
