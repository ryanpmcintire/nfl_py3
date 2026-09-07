# Conditional integer-margin mapping: lane K predeclaration

Frozen before scoring, 2026-09-07. Family `mod18_conditional_margin_v1`.
Read: active manifest `artifacts/active_ats_model.json` identifies a4c757efd2525da6,
weak_stack ridge alpha 10, gaussian_median. Comparator is its own archived opener
probability-rule picks; active forecast and played three-member union stay unchanged.

K1 `conditional_margin_lattice`: Gaussian kernel weights in PREDICTED-MARGIN space,
bandwidth 2.5 points fixed. Count weighted ACTUAL integer margins, with no residual
translation or smoothing between margin integers. Conditioning centers include the
prior residual median, matching the incumbent's location decision. History is a
strict weekly out-of-time prediction stream; use trailing five seasons including
completed earlier weeks of the target season, never the target week. All history
must precede the first game of the target week (dates plus conservative one-day
completion allowance). No future outcome or same-week game enters history.
No history raises an error; no Gaussian fallback. Kernel underflow uses nearest
predicted-margin distance as a numerical offset, preserving relative weights.

K2 `conditional_margin_lattice_keyshift`: same weighted empirical MARGIN lattice
as K1; translate every integer atom by round(target center minus weighted lower
median). This is the nearest-integer median match: a discrete median cannot in
general equal a noninteger predicted margin. It preserves all atom weights and
integer spacing, but can displace absolute key-number peaks. Explicitly measure
push calibration at absolute lines 3 and 7; MOD-05's recentering push defect is a
design constraint to expose, not suppress. No tuning or interpolation after scoring.

K3 `conditional_margin_lattice_keyside`: additionally match the nearest absolute
key number from (3,7,10,14) and sign(abs(line)-key), tie toward smaller key.
Use the cell only with at least 50 prior games and Kish effective sample size >=30
under the fixed kernel; otherwise fall back to K1. These support floors are fixed
regularization choices, not measured optimal constants. Record fallback counts.

The support/counting primitive is MOD-05's empirical weighted integer lattice;
reuse its effective_sample_size utility and key-number frequency diagnostic.
Strict cover = mass(margin>line), push = mass(margin=line), loss = mass(margin<line).
Decision probability = strict cover + 0.5*push (half credit for a pool push).
Read: clv.py's opener grading excludes pushes from accuracy. Thus primary paired
accuracy and Brier/log loss use the same non-push rows; additionally report proper
three-outcome probabilities and push calibration. No smooth pooled-residual read
is substituted for any conditional candidate.

Measurement: identical full 1,537-row opener archive, 1,503 non-push expected;
week-block bootstrap 20,000 draws seed 20260817, within-week correlation ZERO.
Frozen-pick within-week outcome permutation null: 2,000 draws, same seed, report
all arms and maximum-arm null. Brier and log-loss improvement use baseline loss
minus candidate loss so positive always favors the candidate, with own units.
Reliability by [0,3], (3,6.5], (6.5,7.5), [7.5,10], (10,infinity), assigning
7.5 to the latter bucket to avoid overlapping counts; also show the alternate
7.5 allocation. Positive nflverse home spread means HOME FAVOURED.
Report all/favourite/underdog cells, seasons, and exact-key push calibration.
Run overlay-composition CLI on EACH scratch opener artifact, recompute the
coach-fade/division-revenge/player-arrests OR union for candidate and incumbent,
and report paired differences without searching subsets. Week 1 is read-only.

Declare and assign via rotation before scoring. Assign successive two-season
windows only after recording the predecessor; report all arms on each and the
full-archive descriptive union. Record all three metrics and composed accuracy
through weak-signals CLI. Record each assigned look through rotation record.
Ranking: maximum opener accuracy point improvement, ties K1 then K2 then K3;
composed-card EV is the decision read; thresholds only govern claims.
Mined-archive discount: inherited active-model selection, prior spread diagnosis,
prior lattice experiments, three correlated arms, and overlay subset selection
on shared seasons make this descriptive reuse, not independent confirmation.
Do not invent a numerical discount. No activation or production artifact writes.

## Closing-grounds taxonomy (verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close an experiment. At this
evaluator's ~2-point resolution, "contains zero" is the EXPECTED outcome for a real small signal. Only two
grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive control proven able to
detect an effect that size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the binary "contains zero". The
registry code hard-rejects inadmissible closures; if a record command errors, the verdict is wrong, not
the validator. Never use 95%, 0.90, or any threshold as a DECISION bar; decide on expected value
(`probability_positive` above 0.5 favours playing it), thresholds only govern what docs may CLAIM.
Grade at the OPENER (the pool's grade); a close-graded number may never veto a play. Within-week game
correlation is ZERO by owner mandate: never estimate or pad it. Never say something "needs N more
games": the data is fixed and the project is model-limited.
