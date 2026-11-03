import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, HuberRegressor, Ridge, RidgeCV
from sklearn.model_selection import KFold

ALPHA = 10.0
ALPHA_GRID = tuple(10.0 ** (-1 + 0.5 * i) for i in range(13))
FOLDS = 5
HALFLIFE_GRID = (1.0, 2.0, 4.0, 8.0)
WINDOW_GRID = (2, 4, 8)
EPS_GRID = (1.1, 1.35, 1.75, 2.5, 5.0)
CLIP_Q_GRID = (0.90, 0.95, 0.98, 1.0)
L1_GRID = (0.2, 0.5, 0.8)


def sd(m, a, y):
    return float(np.std(y - m.predict(a), ddof=1))


def sign_acc(pred, y):
    keep = y != 0
    return float(np.mean(np.sign(pred[keep]) == np.sign(y[keep])))


def folds(n):
    return list(KFold(FOLDS).split(np.arange(n)))


def pick(candidates, score):
    best, best_s = None, -np.inf
    for c in candidates:
        s = score(c)
        if s > best_s:
            best, best_s = c, s
    return best


def cv_acc(fit_predict, a, y):
    hits = []
    for tr, te in folds(len(y)):
        hits.append(sign_acc(fit_predict(a[tr], y[tr], a[te], tr), y[te]) * len(te))
    return sum(hits) / len(y)


def season_holdout(train):
    s = train["season"].to_numpy()
    last = s.max()
    return np.flatnonzero(s < last), np.flatnonzero(s == last)


def age_years(train):
    s = train["season"].to_numpy(float)
    return s.max() - s


def recency(a, y, b, train):
    age = age_years(train)
    tr, te = season_holdout(train)

    def score(h):
        w = 0.5 ** (age / h)
        m = Ridge(alpha=ALPHA).fit(a[tr], y[tr], sample_weight=w[tr])
        return sign_acc(m.predict(a[te]), y[te])

    h = pick(HALFLIFE_GRID, score)
    w = 0.5 ** (age / h)
    m = Ridge(alpha=ALPHA).fit(a, y, sample_weight=w)
    return m.predict(b), sd(m, a, y)


def window(a, y, b, train):
    s = train["season"].to_numpy()
    tr, te = season_holdout(train)

    def score(k):
        keep = tr[s[tr] > s[te].max() - 1 - k]
        if len(keep) < a.shape[1] * 2:
            return -np.inf
        m = Ridge(alpha=ALPHA).fit(a[keep], y[keep])
        return sign_acc(m.predict(a[te]), y[te])

    k = pick(WINDOW_GRID, score)
    keep = np.flatnonzero(s > s.max() - k)
    m = Ridge(alpha=ALPHA).fit(a[keep], y[keep])
    return m.predict(b), sd(m, a, y)


def huber(a, y, b, train):
    def fp(eps):
        def f(xa, ya, xb, idx):
            return HuberRegressor(epsilon=eps, alpha=ALPHA, max_iter=500).fit(xa, ya).predict(xb)

        return f

    eps = pick(EPS_GRID, lambda e: cv_acc(fp(e), a, y))
    m = HuberRegressor(epsilon=eps, alpha=ALPHA, max_iter=500).fit(a, y)
    return m.predict(b), sd(m, a, y)


def winsor(a, y, b, train):
    def clip_at(ytr, q):
        if q >= 1.0:
            return ytr
        c = np.quantile(np.abs(ytr), q)
        return np.clip(ytr, -c, c)

    def fp(q):
        def f(xa, ya, xb, idx):
            return Ridge(alpha=ALPHA).fit(xa, clip_at(ya, q)).predict(xb)

        return f

    q = pick(CLIP_Q_GRID, lambda c: cv_acc(fp(c), a, y))
    yc = clip_at(y, q)
    m = Ridge(alpha=ALPHA).fit(a, yc)
    return m.predict(b), sd(m, a, y)


def enet(a, y, b, train):
    m = ElasticNetCV(l1_ratio=list(L1_GRID), cv=KFold(FOLDS), alphas=20, max_iter=5000).fit(a, y)
    return m.predict(b), sd(m, a, y)


def hgb(a, y, b, train):
    m = HistGradientBoostingRegressor(early_stopping=True, random_state=0).fit(a, y)
    return m.predict(b), sd(m, a, y)


def hgb_resid(a, y, b, train):
    r = Ridge(alpha=ALPHA).fit(a, y)
    res = y - r.predict(a)
    g = HistGradientBoostingRegressor(early_stopping=True, random_state=0).fit(a, res)
    pred = r.predict(b) + g.predict(b)
    return pred, float(np.std(res - g.predict(a), ddof=1))


def products(x):
    n = x.shape[1]
    ix = [x[:, i] * x[:, j] for i in range(n) for j in range(i + 1, n)]
    return np.column_stack([x, *ix])


def ix_a10(a, y, b, train):
    pa, pb = products(a), products(b)
    m = Ridge(alpha=ALPHA).fit(pa, y)
    return m.predict(pb), sd(m, pa, y)


def ix_cv(a, y, b, train):
    pa, pb = products(a), products(b)
    m = RidgeCV(alphas=ALPHA_GRID[:9], cv=KFold(FOLDS)).fit(pa, y)
    return m.predict(pb), sd(m, pa, y)


def accalpha(a, y, b, train):
    def fp(al):
        def f(xa, ya, xb, idx):
            return Ridge(alpha=al).fit(xa, ya).predict(xb)

        return f

    al = pick(ALPHA_GRID, lambda c: cv_acc(fp(c), a, y))
    m = Ridge(alpha=al).fit(a, y)
    return m.predict(b), sd(m, a, y)


def slope(a, y, b, train):
    oof = np.zeros(len(y))
    for tr, te in folds(len(y)):
        oof[te] = Ridge(alpha=ALPHA).fit(a[tr], y[tr]).predict(a[te])
    k = float(oof @ y / (oof @ oof))
    m = Ridge(alpha=ALPHA).fit(a, y)
    return k * m.predict(b), sd(m, a, y)
