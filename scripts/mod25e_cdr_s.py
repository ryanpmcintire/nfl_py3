import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import mod25e_endgame as eg

for d in sys.argv[1:]:
    S = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(d + "/play_*_*.parquet"))], ignore_index=True)
    S = S[(S.qtr == 4) & (S.gsr <= 300) & (S.sd <= 0) & (S.sd >= -2) & (S.yl <= 40) & (S.down <= 3) & S.code.isin(eg.LATE_CODES)]
    z = S[S.down >= 2]
    kick = S[S.code == 3]
    print(d, "n states d2-3", len(z), "P(gsr<=5)", round((z.gsr <= 5).mean(), 3), "n kicks", len(kick), "late kick share", round((kick.gsr <= 5).mean(), 3), "kicks gsr<=5 / states", round((S.code == 3).mul(S.gsr <= 5).sum() / len(S), 4))
