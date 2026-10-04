import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod25e_crH as H  # noqa: E402
import mod25e_kfit as kf  # noqa: E402

orig = H.run_with_flags


def wrapped(init, setting):
    orig(init, setting)
    if kf.enabled():
        kf.install_kf()


H.run_with_flags = wrapped

if __name__ == "__main__":
    H.main()
