import os
from pathlib import Path

NV = Path(
    os.environ.get(
        "NFL_PBP_FULL_DIR",
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "nflverse_pbp_full"
        / "20261103T131922Z",
    )
)
