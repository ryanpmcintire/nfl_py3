# MOD-56 leader-book depth in the move term

Goal: decide whether the move term should depend on how many leader books back it (CLE_NO served 2.5 from 2 books).

State: Measured. Q1 cause: serve window needs a quote before Wed 00:00 ET (sharp_book_movement_features.py:121 observed>=_monday, :139 _first_as_of<_wednesday) and a post-Wed diff. Wk5 2026 leader capture began Wed 10-07 ~18:00 ET (no Mon/Tue quotes), bovada/mybookie fail :139; williamhill_us has Aug-Sep and Fri quotes only, move NaN. Wk2: no quotes Wed-Sat (Sun excluded by :122). Both gave leader_books=0, move 0. leader_books<=2 in 2026: wk2 (16), wk5 (15), wk7 SF_ATL, wk8 IND_JAX LV_NYJ, wk9 17 of 18 at 2 (HOU_LAC 1) because williamhill_us has no Mon-Wed quotes.
Q2: fit population has exactly 3 leader books in all 799 games, so no depth variation exists to bucket. Simulated k=2 (3 pairs). Per-fold tau2 0.99-1.19, sigma_b2 0.14-0.16, reliability r3 0.953-0.960, r2 0.932-0.941, w2 0.977-0.980. Variant vs current held-out LL diff +1.3e-5 [-1.7e-4,+1.9e-4] p_pos 0.557; no side differs. Move coef per fold 0.229/0.245/0.189. LOSO LL k=3 0.68342 vs in-sample 0.68631 (model-only 0.69729); weekly top pick 32 of 54. Depth noise is not the lever: signal variance dominates, so 2.5 points at 0.22 logit/pt is simply the linear term.
Recorded: mod56_leader_book_depth_precision_weight, unresolved_below_power.
Tried: precision weight (1 arm), agreement-range buckets (3 looks), |move| buckets (2 looks).
Next: bound or saturate nothing without a named mechanism; test move-size nonlinearity at the discrete key numbers, and fix williamhill_us Mon-Wed capture gap.
Open: why williamhill_us has no Mon-Wed quotes in 2026 wk3-9.
Scripts: tests/scratch/mod56_q1b.py, mod56_q1c.py, mod56_fit.py
