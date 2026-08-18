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

Subagents never see the session hooks or this file's context injection. Any
subagent prompt that runs, scores, or adjudicates an experiment MUST paste the
closing-grounds taxonomy above verbatim, and verdicts must flow through
`nfl-ats weak-signals record` / `nfl-ats rotation record-look` — never through
prose in a doc.

@AGENTS.md
