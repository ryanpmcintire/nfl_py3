import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
BASE_STATE = REPO / "artifacts" / "sim10"
FIRST_SEASON = 2009
ARMS = ("ridge_nomkt", "hgb_nomkt")
TEST_SEASONS = tuple(range(2020, 2026))
STAGES = ("state", "produce", "truth", "features", "student2")


def run(argv, env, log):
    t0 = time.time()
    with open(log, "a") as fh:
        fh.write("$ " + " ".join(str(a) for a in argv) + "\n")
        fh.flush()
        r = subprocess.run(
            [sys.executable, *[str(a) for a in argv]],
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=REPO,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    if r.returncode != 0:
        raise SystemExit(f"failed {argv} see {log}")
    return time.time() - t0


def base_env():
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    return env


def state_env(T, sdir):
    env = base_env()
    env["SIM10_OUT"] = str(sdir)
    env["SIM10_SEASONS"] = f"{FIRST_SEASON}-{T - 1}"
    env["SIM11_OUT"] = str(sdir)
    env["MOD25_STRENGTH"] = str(sdir / "strength_terms.json")
    env["MOD25_BANDWIDTHS"] = str(sdir / "bandwidths_week.json")
    env["MOD25_LATENT"] = str(sdir / "latent_fit.json")
    env["MOD25_Q4DEC"] = str(sdir / "q4dec_fg.json")
    return env


def stage_state(T, sdir, timing, log):
    sdir.mkdir(parents=True, exist_ok=True)
    for f in ("hgb_params.json", "bandwidths_week.json"):
        if not (sdir / f).exists():
            shutil.copy(BASE_STATE / f, sdir / f)
    env = state_env(T, sdir)
    drive = SCRIPTS / "sim10_drive.py"
    q4 = sdir / "q4dec.json"
    steps = (
        ("build", [drive, "build"], "csr.npz"),
        ("outfit", [drive, "outfit"], "outcome_models.pkl"),
        ("patfit", [drive, "patfit"], "pat_model.pkl"),
        (
            "latfit",
            [
                drive,
                "latfit",
                "--seasons",
                f"{FIRST_SEASON}-{T - 1}",
                "--out",
                sdir / "latent_fit.json",
            ],
            "latent_fit.json",
        ),
        (
            "strfit",
            [SCRIPTS / "sim11_strength.py", "strfit", "--out", sdir / "strength_terms.json"],
            "strength_terms.json",
        ),
        (
            "q4dec",
            [
                drive,
                "q4dec",
                "--out",
                q4,
                "--h",
                sdir / "bandwidths_week.json",
                "--strength",
                sdir / "strength_terms.json",
                "--latent",
                sdir / "latent_fit.json",
                "--q4dec",
                "",
            ],
            "q4dec.json",
        ),
    )
    for name, argv, product in steps:
        if (sdir / product).exists():
            continue
        timing[f"state_{name}"] = run(argv, env, log)
    fg = json.loads(q4.read_text())
    (sdir / "q4dec_fg.json").write_text(json.dumps({"td": {}, "fg": fg["fg"]}, indent=1))


def stage_cmd(stage, T, sdir, syn, out, a):
    script = SCRIPTS / "mod25_v3_produce.py"
    head = [script, "--syn", syn, "--out", out]
    if stage == "produce":
        return [
            *head,
            "produce",
            "--worlds",
            a.worlds,
            "--seasons",
            a.seasons,
            "--workers",
            a.workers,
            "--chunks",
            a.chunks,
        ]
    if stage == "features":
        return [*head, "features", "--workers", a.workers]
    return [*head, stage]


def run_T(T, a, timing):
    sdir = Path(a.state_root).resolve() / f"T{T}"
    syn = Path(a.syn_root).resolve() / f"T{T}"
    out = Path(a.pipe_root).resolve() / f"T{T}"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "wf.log"
    for stage in a.stages.split(","):
        if stage == "state":
            stage_state(T, sdir, timing, log)
            continue
        timing[stage] = run(stage_cmd(stage, T, sdir, syn, out, a), state_env(T, sdir), log)
    (out / "wf_timing.json").write_text(json.dumps(timing, indent=1))


def cmd_join(a):
    ts = [int(t) for t in a.ts.split(",")]
    pipe = Path(a.pipe_root).resolve()
    frames = {t: pd.read_parquet(pipe / f"T{t}" / "real_distilled.parquet") for t in ts}
    base = frames[min(ts)]
    cols = [c for c in base.columns if c not in ("game_id", "season")]
    parts = []
    owner = {}
    for s in sorted(base["season"].unique()):
        cands = [t for t in ts if t <= s]
        owner[int(s)] = max(cands) if cands else min(ts)
    for s, t in owner.items():
        f = frames[t]
        parts.append(f[f["season"] == s])
    out = pd.concat(parts, ignore_index=True)[["game_id", "season", *cols]]
    pipe.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pipe / "real_distilled.parquet")
    (pipe / "wf_owner.json").write_text(json.dumps(owner, indent=1))
    print(json.dumps(owner))


def cmd_grade(a):
    env = base_env()
    pipe = Path(a.pipe_root).resolve()
    for arm in a.arms.split(","):
        t = run(
            [SCRIPTS / "mod25_v3_produce.py", "--out", pipe, "grade", "--arm", arm],
            env,
            pipe / "wf.log",
        )
        print(arm, f"{t:.0f}s", flush=True)


def cmd_run(a):
    for T in (int(t) for t in a.ts.split(",")):
        timing = {}
        run_T(T, a, timing)
        print(f"T{T}", json.dumps(timing), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ts_default = ",".join(str(t) for t in TEST_SEASONS)
    r = sub.add_parser("run")
    r.add_argument("--ts", default=ts_default)
    r.add_argument("--stages", default=",".join(STAGES))
    r.add_argument("--worlds", type=int, default=6)
    r.add_argument("--seasons", type=int, default=10)
    r.add_argument("--chunks", type=int, default=8)
    r.add_argument("--workers", type=int, default=6)
    j = sub.add_parser("join")
    j.add_argument("--ts", default=ts_default)
    g = sub.add_parser("grade")
    g.add_argument("--arms", default=",".join(ARMS))
    for sp in (r, j, g):
        sp.add_argument("--state-root", default=str(REPO / "artifacts" / "sim10_wf"))
        sp.add_argument("--syn-root", default=str(REPO / "data" / "processed" / "synthetic" / "wf"))
        sp.add_argument("--pipe-root", default=str(REPO / "artifacts" / "sim10_wf" / "pipeline"))
    a = ap.parse_args()
    {"run": cmd_run, "join": cmd_join, "grade": cmd_grade}[a.cmd](a)


if __name__ == "__main__":
    main()
