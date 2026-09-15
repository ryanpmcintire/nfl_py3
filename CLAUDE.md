# nfl_py3 session contract

The binding repository instructions live in `AGENTS.md`, imported below. They
are owner mandates, not suggestions, and they apply to every session and every
model tier. This file exists because sessions were observed skipping AGENTS.md
entirely; the import guarantees it loads.

The single most-violated rule, restated here so it survives even a failed
import: **an interval or CI that contains zero is NEVER grounds to reject,
fail, or close an experiment.** At this evaluator's ~2-point resolution,
"contains zero" is the EXPECTED outcome for a real small signal. Only two
grounds ever close a line of work: (1) refuted mechanism — a RESOLVED wrong
sign (whole interval on the wrong side of zero) or zero split-half
reliability; (2) bounded by a positive control proven able to detect an
effect that size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible closures;
if a record command errors, the verdict is wrong, not the validator.

The second rule, equally binding and the opposite failure: **the rule above
never promotes anything.** One calibrated probability decides every pick. No
rule changes a served side on its own signal; every situational signal is a
fitted term in the model's probability, every weight or threshold is chosen
leave-one-season-out with the in-sample number reported beside it, calibration
is reported alongside hit rate, small lopsided splits get a permutation null,
and every look is counted. A member that can only flip a side is not a signal.

Subagents never see the session hooks or this file's context injection. Any
subagent prompt that runs, scores, or adjudicates an experiment MUST paste the
closing-grounds taxonomy above AND the one-probability rule verbatim, and
verdicts must flow through
`nfl-ats weak-signals record` / `nfl-ats rotation record-look` — never through
prose in a doc.

@AGENTS.md
