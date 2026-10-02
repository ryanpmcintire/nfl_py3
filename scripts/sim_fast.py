import hashlib
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "data" / "processed" / "sim_cache"


def enabled(cfg=None):
    if cfg is not None and cfg.get("fast"):
        return True
    return os.environ.get("SIM_FAST", "0") == "1"


class FastGBM:
    def __init__(self, clf):
        preds = clf._predictors
        self.K = int(clf.n_trees_per_iteration_)
        self.iters = len(preds)
        trees = [p for it in preds for p in it]
        T = len(trees)
        M = max(len(p.nodes) for p in trees)
        self.feat = np.zeros((T, M), dtype=np.int64)
        self.thr = np.zeros((T, M), dtype=np.float64)
        self.left = np.zeros((T, M), dtype=np.int64)
        self.right = np.zeros((T, M), dtype=np.int64)
        self.leaf = np.ones((T, M), dtype=bool)
        self.mgl = np.zeros((T, M), dtype=bool)
        self.val = np.zeros((T, M), dtype=np.float64)
        depth = 0
        for t, p in enumerate(trees):
            n = p.nodes
            m = len(n)
            if n["is_categorical"].any():
                raise ValueError("categorical splits unsupported")
            self.feat[t, :m] = n["feature_idx"]
            self.thr[t, :m] = n["num_threshold"]
            self.left[t, :m] = n["left"]
            self.right[t, :m] = n["right"]
            self.leaf[t, :m] = n["is_leaf"].astype(bool)
            self.mgl[t, :m] = n["missing_go_to_left"].astype(bool)
            self.val[t, :m] = n["value"]
            depth = max(depth, int(n["depth"].max()))
        self.depth = depth
        self.T = T
        self.ar = np.arange(T)
        base = np.asarray(clf._baseline_prediction, dtype=np.float64).reshape(-1)
        self.base = base if base.size == self.K else np.full(self.K, float(base[0]))
        self.classes = np.asarray(getattr(clf, "classes_", [0]))

    def raw(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        node = np.zeros(self.T, dtype=np.int64)
        ar = self.ar
        for _ in range(self.depth):
            f = self.feat[ar, node]
            xv = x[f]
            go_left = np.where(np.isnan(xv), self.mgl[ar, node], xv <= self.thr[ar, node])
            nxt = np.where(go_left, self.left[ar, node], self.right[ar, node])
            node = np.where(self.leaf[ar, node], node, nxt)
        v = self.val[ar, node].reshape(self.iters, self.K)
        stack = np.concatenate([self.base[None, :], v], axis=0)
        return np.cumsum(stack, axis=0)[-1]

    def raw_batch(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        out = np.empty((n, self.K), dtype=np.float64)
        ar = self.ar[None, :]
        for s in range(0, n, 2000):
            xb = X[s : s + 2000]
            m = xb.shape[0]
            node = np.zeros((m, self.T), dtype=np.int64)
            for _ in range(self.depth):
                f = self.feat[ar, node]
                xv = np.take_along_axis(xb, f, axis=1)
                go_left = np.where(np.isnan(xv), self.mgl[ar, node], xv <= self.thr[ar, node])
                nxt = np.where(go_left, self.left[ar, node], self.right[ar, node])
                node = np.where(self.leaf[ar, node], node, nxt)
            v = self.val[ar, node].reshape(m, self.iters, self.K)
            stack = np.concatenate([np.broadcast_to(self.base, (m, 1, self.K)), v], axis=1)
            out[s : s + m] = np.cumsum(stack, axis=1)[:, -1, :]
        return out

    def label(self, x):
        r = self.raw(x)
        enc = int(r[0] > 0) if self.K == 1 else int(np.argmax(r))
        return int(self.classes[enc])


_FAST_CLF = {}


def fast_gbm_predict_label(clf, bitsets, feat):
    g = _FAST_CLF.get(id(clf))
    if g is None or g[0] is not clf:
        g = (clf, FastGBM(clf))
        _FAST_CLF[id(clf)] = g
    return g[1].label(feat[0])


class Unkeyable(Exception):
    pass


_FILE_SIG = {}


def _file_sig(path):
    p = Path(path)
    st = p.stat()
    k = (str(p), st.st_size, st.st_mtime_ns)
    v = _FILE_SIG.get(k)
    if v is None:
        h = hashlib.sha1()
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 22)
                if not b:
                    break
                h.update(b)
        v = h.hexdigest()
        _FILE_SIG[k] = v
    return v


def _sig_index():
    return CACHE_DIR / "_filehash.json"


def _persisted_file_sig(path):
    import json

    p = Path(path)
    st = p.stat()
    k = f"{p}|{st.st_size}|{st.st_mtime_ns}"
    idx = _sig_index()
    try:
        d = json.loads(idx.read_text())
    except Exception:
        d = {}
    if k in d:
        return d[k]
    v = _file_sig(p)
    d[k] = v
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = idx.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, idx)
    except OSError:
        pass
    return v


def _consts(mod):
    out = []
    for k in sorted(vars(mod)):
        if k.isupper():
            v = getattr(mod, k)
            if isinstance(v, (int, float, str, bool, tuple)) or v is None:
                out.append((k, repr(v)))
    return out


_ENV_SIG = {}


def _env_sig(sim, mods):
    h = hashlib.sha1()
    for m in mods:
        h.update(_persisted_file_sig(m.__file__).encode())
    h.update(repr(_consts(sim)).encode())
    snap = Path(sim.PBP_SNAPSHOT_DIR)
    for p in sorted(snap.glob("season=*/plays.parquet")):
        h.update(str(p.parent.name).encode())
        h.update(_persisted_file_sig(p).encode())
    gf = Path(sim.GAME_FEATURES_PATH)
    if gf.exists():
        h.update(_persisted_file_sig(gf).encode())
    return h.hexdigest()


def _key_of(o):
    import pandas as pd

    if o is None or isinstance(o, (bool, int, float, str, np.integer, np.floating)):
        return repr(o)
    if isinstance(o, (tuple, list)):
        return "[" + ",".join(_key_of(x) for x in o) + "]"
    if isinstance(o, dict):
        return "{" + ",".join(f"{_key_of(k)}:{_key_of(o[k])}" for k in sorted(o, key=repr)) + "}"
    if isinstance(o, np.ndarray):
        if o.dtype == object:
            raise Unkeyable
        return "nd" + hashlib.sha1(np.ascontiguousarray(o).tobytes()).hexdigest() + str(o.shape) + str(o.dtype)
    if isinstance(o, pd.DataFrame):
        k = o.attrs.get("sf_key")
        if k is not None and o.attrs.get("sf_len") == len(o):
            return "df" + k
        try:
            hv = pd.util.hash_pandas_object(o, index=True).to_numpy()
        except Exception as e:
            raise Unkeyable from e
        return "dfh" + hashlib.sha1(hv.tobytes()).hexdigest() + "|".join(map(str, o.columns))
    raise Unkeyable


def _memo(fn, modname, ctx):
    import pandas as pd

    if getattr(fn, "_sf_memo", False):
        return fn

    def wrapped(*args, **kwargs):
        try:
            sig = ctx["sig"]()
            key = hashlib.sha1((modname + "." + fn.__name__ + "|" + sig + "|" + _key_of(list(args)) + "|" + _key_of(kwargs)).encode()).hexdigest()
        except Unkeyable:
            return fn(*args, **kwargs)
        path = CACHE_DIR / f"{fn.__name__}_{key[:24]}.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        res = fn(*args, **kwargs)
        if isinstance(res, pd.DataFrame):
            res.attrs["sf_key"] = key
            res.attrs["sf_len"] = len(res)
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            with open(tmp, "wb") as f:
                pickle.dump(res, f, protocol=5)
            os.replace(tmp, path)
        except OSError:
            pass
        return res

    wrapped._sf_memo = True
    wrapped.__name__ = fn.__name__
    wrapped.__wrapped__ = fn
    return wrapped


MEMO_TARGETS = (
    ("sim", "load_reg_seasons"),
    ("sim", "load_team_ratings"),
    ("sim", "build_transition_frame"),
    ("sim", "build_neighbor_index"),
    ("sim", "build_neighbor_index_scipy"),
    ("sim", "fit_fourth_down_policy"),
    ("sim", "build_fourth_down_group_index"),
    ("sim", "build_opening_pool"),
    ("m25", "fit_decisions"),
    ("m25", "fit_pat"),
    ("m25", "build_class_trees"),
    ("c25", "attrs_from"),
    ("c25", "build_trees_flag"),
    ("c25", "build_ot_pool"),
    ("c25", "clock_table"),
    ("dv", "blup_rows"),
    ("dv", "ipw_fit"),
    ("dv", "rate_effects"),
    ("dv", "rate_defs"),
)


def _mods():
    import mod25_mechanisms as m25
    import mod25c_noise as c25
    import mod25d_variance as dv
    import mod25_generator as gen
    import sim04_engine as sim

    return {"sim": sim, "m25": m25, "c25": c25, "dv": dv, "gen": gen}


def _parts():
    return set(os.environ.get("SIM_FAST_PARTS", "memo,gbm,fm,ep").split(","))


def activate():
    if "memo" not in _parts():
        return
    if "LOKY_MAX_CPU_COUNT" not in os.environ:
        os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)
    M = _mods()
    sim = M["sim"]
    files = [M[k] for k in ("sim", "m25", "c25", "dv", "gen")]
    cache = {}

    def sig():
        k = (str(sim.PBP_SNAPSHOT_DIR), repr(_consts(sim)))
        v = cache.get(k)
        if v is None:
            v = _env_sig(sim, files)
            cache.clear()
            cache[k] = v
        return v

    ctx = {"sig": sig}
    for mk, name in MEMO_TARGETS:
        mod = M[mk]
        fn = getattr(mod, name, None)
        if fn is not None:
            setattr(mod, name, _memo(fn, mod.__name__, ctx))


def feature_matrix_fast(orig, sim):
    late = tuple(sim.LATE_PHASES)

    def fm(dist, fp, score, time_raw, off_to, def_to, phase_arr):
        try:
            if (
                np.ndim(dist) == 1
                and len(dist) == 1
                and len(fp) == 1
                and len(score) == 1
                and len(time_raw) == 1
                and len(off_to) == 1
                and len(def_to) == 1
                and np.ndim(phase_arr) == 1
                and len(phase_arr) == 1
            ):
                d, f, s, t, o, e = float(dist[0]), float(fp[0]), float(score[0]), float(time_raw[0]), float(off_to[0]), float(def_to[0])
                if math.isfinite(d) and math.isfinite(f) and math.isfinite(s) and math.isfinite(t) and math.isfinite(o) and math.isfinite(e):
                    tw = 1.0 if phase_arr[0] in late else 0.0
                    clip = sim.SCORE_CLIP
                    sc = min(max(s, -clip), clip)
                    a = abs(sc)
                    mag = min(a, sim.SCORE_INNER) / sim.SCORE_INNER_SCALE + max(a - sim.SCORE_INNER, 0.0) / sim.SCORE_OUTER_SCALE
                    sg = 1.0 if sc > 0 else (-1.0 if sc < 0 else 0.0)
                    return np.array(
                        [[d / sim.SCALE_YDSTOGO, f / sim.SCALE_FP, sg * mag, t / sim.SCALE_TIME, (o / sim.SCALE_TIMEOUTS) * tw, (e / sim.SCALE_TIMEOUTS) * tw]],
                        dtype=np.float64,
                    )
        except TypeError:
            pass
        return orig(dist, fp, score, time_raw, off_to, def_to, phase_arr)

    return fm


class FastReg:
    def __init__(self, reg):
        self.reg = reg
        self.fg = FastGBM(reg)

    def predict(self, X):
        return self.fg.raw_batch(X)[:, 0]

    def __getattr__(self, name):
        return getattr(self.reg, name)


def finish():
    M = _mods()
    sim = M["sim"]
    G = M["c25"]._G
    ns = G.get("ns")
    if ns is None:
        return
    if "gbm" in _parts():
        ns["fast_gbm_predict_label"] = fast_gbm_predict_label
    if "fm" in _parts():
        ns["feature_matrix"] = feature_matrix_fast(sim.feature_matrix, sim)
    if "ep" in _parts() and "ep_model" in G and not isinstance(G["ep_model"], FastReg):
        G["ep_model"] = FastReg(G["ep_model"])
