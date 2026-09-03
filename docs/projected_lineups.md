# Projected lineups on This Week

The This Week game deep dive includes an optional **Projected lineups & model
impact** panel. It is a static view designed for GitHub Pages: a refresh job
builds `artifacts/lineups/<stamp>/lineups.json`, and the renderer only reads
that artifact. The artifact is ignored by Git along with the other local data
and model outputs.

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
