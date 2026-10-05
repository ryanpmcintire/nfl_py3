import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import sim09_f2 as f2  # noqa: E402

LABEL = "crHpqokgndecsmfwtjo2as2"
ART = REPO / "artifacts" / "mod25e3"
OUT = []


def say(s=""):
    print(s)
    OUT.append(str(s))


def main():
    pool = pd.read_parquet(REPO / "artifacts" / "sim09" / "u4g" / "pool.parquet")
    say(f"pool.parquet rows {len(pool)} cols {list(pool.columns)}")
    dv = f2.dv
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    tr = dv.sim.build_transition_frame(pbp)
    say(f"E92 rebuilt transition frame rows {len(tr)}")
    ep = pd.concat([pd.read_parquet(f"{dv.m25.SNAP}/season={y}/plays.parquet", columns=["game_id", "play_id", "play_type", "penalty", "penalty_yards"]) for y in dv.TRAIN]).drop_duplicates(["game_id", "play_id"])
    m = tr[["game_id", "play_id", "yards_gained"]].merge(ep[["game_id", "play_id", "play_type", "penalty", "penalty_yards"]], on=["game_id", "play_id"], how="left")
    pen = (m.penalty == 1).to_numpy()
    say(f"rebuilt frame: penalty rate {pen.mean():.4f}; no_play share {(m.play_type == 'no_play').mean():.4f}; no_play&penalty {((m.play_type == 'no_play') & pen).mean():.4f}")
    say(f"rebuilt frame yards_gained: penalty rows mean {m.yards_gained[pen].mean():.2f} sd {m.yards_gained[pen].std():.2f}; nonpen mean {m.yards_gained[~pen].mean():.2f}; penalty rows with yards_gained==0 {(m.yards_gained[pen] == 0).mean():.3f}; mean penalty_yards on penalty rows {m.penalty_yards[pen].mean():.2f}")
    if len(pool) == len(tr):
        say(f"key alignment pool gid/pid vs tr game_id/play_id mismatches {int(((pool.gid.to_numpy() != tr.game_id.to_numpy()) | (pool.pid.to_numpy() != tr.play_id.to_numpy())).sum())}")
        pool["yards_gained"] = tr.yards_gained.to_numpy()
        for c in ("penalty", "yards_gained", "play_type_code"):
            if c in pool.columns and c in m.columns:
                pass
        if "yards_gained" in pool.columns:
            say(f"pool vs rebuilt yards_gained max abs diff {np.nanmax(np.abs(pool.yards_gained.to_numpy() - m.yards_gained.to_numpy())):.4f}")
        if "penalty" in pool.columns:
            say(f"pool vs rebuilt penalty flag mismatches {int((pool.penalty.fillna(0).to_numpy() == 1).astype(int).__ne__(pen.astype(int)).sum())}")
    else:
        say(f"LENGTH MISMATCH pool {len(pool)} vs rebuilt {len(tr)}")
    pp = (pool.penalty.fillna(0) == 1).to_numpy() if "penalty" in pool.columns else None
    if pp is not None:
        say(f"pool.parquet penalty rate {pp.mean():.4f}")
        if "yards_gained" in pool.columns:
            say(f"pool.parquet penalty-row yards_gained mean {pool.yards_gained[pp].mean():.2f} sd {pool.yards_gained[pp].std():.2f} nonpen mean {pool.yards_gained[~pp].mean():.2f}")
    cc = pool.code.to_numpy()
    say(f"pool code6 share {(cc == 6).mean():.4f}; pool penalty share among code6 {pp[cc == 6].mean():.3f}; penalty rows not code6 {(pp & (cc != 6)).mean():.4f}; code6 pool mean abs yards_gained {pool.yards_gained[cc == 6].abs().mean():.2f}")
    for sd in (11, 12, 13):
        sdir = ART / f"e5_{LABEL}_s{sd}"
        parts = [pd.read_parquet(f) for f in sorted(sdir.glob("play_*_*.parquet")) if int(f.stem.split("_")[2]) >= 2]
        S = pd.concat(parts, ignore_index=True)
        if sd == 11:
            say(f"play file cols {list(S.columns)}")
        ix = S.idx.fillna(-1).astype(int).to_numpy()
        ok = ix >= 0
        say(f"seed {sd}: plays {len(S)} idx<0 {int((~ok).sum())} idx max {ix.max()} pool len {len(pool)}")
        ixc = np.where(ok, ix, 0)
        spen = pp[ixc] & ok
        say(f"  drawn penalty share {spen.mean():.4f} (pool {pp.mean():.4f}); drawn no_play share {(m.play_type.to_numpy()[ixc] == 'no_play')[ok].mean():.4f}")
        if "yards" in S.columns:
            y = S.yards.to_numpy()
            say(f"  sim play yards: penalty-drawn rows mean {y[spen].mean():.2f} sd {y[spen].std():.2f} share zero {(y[spen] == 0).mean():.3f}; nonpen mean {y[~spen & ok].mean():.2f}")
            if "yards_gained" in pool.columns:
                pg = pool.yards_gained.to_numpy()[ixc]
                say(f"  sim yards vs pool yards_gained on penalty rows: mean diff {(y[spen] - pg[spen]).mean():.3f}; share equal {(y[spen] == pg[spen]).mean():.3f}")
        nx = S.groupby("g")[["yl", "offhome"]].shift(-1)
        sm = (nx.offhome == S.offhome) & (S.flip == 0) & (S.po == 0) & (S.pdf == 0) & S.code.isin([0, 1, 2, 3, 6])
        mv = (S.yl - nx.yl)[sm]
        resid = mv - S.yards[sm]
        say(f"  code6 share of all plays {(S.code == 6).mean():.4f}; same-side moves n {int(sm.sum())}; movement minus recorded yards mean {resid.mean():.3f} mean abs {resid.abs().mean():.3f} share nonzero {(resid.abs() > 0.5).mean():.3f}; code6 rows moved yards mean {mv[S.code[sm] == 6].mean():.2f} abs {mv[S.code[sm] == 6].abs().mean():.2f}")
        if "code" in S.columns:
            say(f"  code counts for penalty-drawn rows {S.code[spen].value_counts().to_dict()}")
    (REPO / "artifacts" / "mod25e3" / "e92" / "e92b.txt").write_text(chr(10).join(OUT))


if __name__ == "__main__":
    main()
