import inspect
import os

SUDDEN_DEATH_LAST_YEAR = 2011
PERIOD_MINUTES_BEFORE_HALVING = 15
PERIOD_MINUTES_AFTER_HALVING = 10
HALVING_FIRST_YEAR = 2017
SECONDS_PER_MINUTE = 60
FIELD_GOAL_ENDS_OLD = "(points_def > 0 or points_off >= 6)"
FIELD_GOAL_ENDS_NEW = "(points_def > 0 or points_off >= (1 if OT_SUDDEN else 6))"


def ot_rules(year):
    minutes = PERIOD_MINUTES_BEFORE_HALVING if year < HALVING_FIRST_YEAR else PERIOD_MINUTES_AFTER_HALVING
    return float(minutes * SECONDS_PER_MINUTE), year <= SUDDEN_DEATH_LAST_YEAR


def era_years(spec):
    lo, hi = (int(x) for x in spec.split("-"))
    return lo, hi


def season_year(spec, sidx):
    lo, hi = era_years(spec)
    return lo + sidx % (hi - lo + 1)


def enabled():
    return bool(os.environ.get("OTY"))


def hook_source():
    orig = inspect.getsource
    seen = {"n": 0}

    def gs(o):
        s = orig(o)
        if getattr(o, "__name__", "") == "run_one_game" and FIELD_GOAL_ENDS_OLD in s:
            s = s.replace(FIELD_GOAL_ENDS_OLD, FIELD_GOAL_ENDS_NEW)
            seen["n"] += 1
        return s

    inspect.getsource = gs
    return orig, seen


def install(dv):
    spec = os.environ["OTY"]
    ns = dv._G["ns"]
    base = ns["run_one_game"]
    cur = {"y": None}

    def run(state, tables, rng, ot_seconds, *rest):
        secs, sudden = ot_rules(cur["y"])
        ns["OT_SUDDEN"] = sudden
        return base(state, tables, rng, secs, *rest)

    ns["OT_SUDDEN"] = False
    ns["run_one_game"] = run
    orig_ps = dv.d_play_season

    def ps(task):
        cur["y"] = season_year(spec, task[1])
        return orig_ps(task)

    dv.d_play_season = ps
