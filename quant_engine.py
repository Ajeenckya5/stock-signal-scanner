"""
Institutional quant stack for long-horizon US + India equities.

Implementations follow the cited papers/textbooks (educational approximations;
not a licensed commercial factor library). Outputs feed BUY/SELL scoring and
the /research dossier.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Research catalog (shown in the UI)
# ---------------------------------------------------------------------------

RESEARCH_CATALOG: List[Dict[str, str]] = [
    {"id": "jt1993", "paper": "Jegadeesh & Titman (1993)", "topic": "12–1 cross-sectional momentum"},
    {"id": "jeg1990", "paper": "Jegadeesh (1990)", "topic": "Short-term reversal"},
    {"id": "amp2013", "paper": "Asness, Moskowitz & Pedersen (2013)", "topic": "Value and momentum everywhere"},
    {"id": "ff1993", "paper": "Fama & French (1993, 2015)", "topic": "Size / value / profitability / investment proxies"},
    {"id": "ahxz2006", "paper": "Ang, Hodrick, Xing & Zhang (2006)", "topic": "Idiosyncratic / low-volatility anomaly"},
    {"id": "fp2014", "paper": "Frazzini & Pedersen (2014)", "topic": "Betting-against-beta (BAB)"},
    {"id": "nm2013", "paper": "Novy-Marx (2013)", "topic": "Gross profitability"},
    {"id": "sharpe1964", "paper": "Sharpe (1964, 1966)", "topic": "CAPM beta / alpha / Sharpe ratio"},
    {"id": "sortino1994", "paper": "Sortino & Price (1994)", "topic": "Downside-deviation Sortino ratio"},
    {"id": "rm1996", "paper": "J.P. Morgan RiskMetrics (1996)", "topic": "EWMA volatility (λ=0.94)"},
    {"id": "engle1982", "paper": "Engle (1982); Bollerslev (1986)", "topic": "GARCH(1,1) variance"},
    {"id": "park1980", "paper": "Parkinson (1980)", "topic": "High–low range volatility"},
    {"id": "gk1980", "paper": "Garman & Klass (1980)", "topic": "OHLC volatility estimator"},
    {"id": "yz2000", "paper": "Yang & Zhang (2000)", "topic": "Overnight + RS volatility"},
    {"id": "amihud2002", "paper": "Amihud (2002)", "topic": "Illiquidity ( |r| / dollar volume )"},
    {"id": "hurst1951", "paper": "Hurst (1951); Mandelbrot", "topic": "R/S Hurst exponent (trend vs mean-reversion)"},
    {"id": "lm1988", "paper": "Lo & MacKinlay (1988)", "topic": "Variance-ratio test"},
    {"id": "df1979", "paper": "Dickey & Fuller (1979)", "topic": "Unit-root / stationarity statistic"},
    {"id": "ou", "paper": "Ornstein–Uhlenbeck; Lo & MacKinlay AR(1)", "topic": "Mean-reversion half-life"},
    {"id": "kelly1956", "paper": "Kelly (1956)", "topic": "Growth-optimal (fractional Kelly) size"},
    {"id": "kaufman1995", "paper": "Kaufman (1995)", "topic": "Efficiency ratio (trend vs chop)"},
    {"id": "hamilton1989", "paper": "Hamilton (1989) — reduced form", "topic": "Two-state volatility regime"},
    {"id": "bns2004", "paper": "Barndorff-Nielsen & Shephard (2004)", "topic": "Jump proxy via |r| vs bipower"},
    {"id": "ru2000", "paper": "Rockafellar & Uryasev (2000)", "topic": "CVaR / expected shortfall"},
    {"id": "vwap1988", "paper": "Berkowitz, Logue & Noser (1988)", "topic": "VWAP as execution / fair-value benchmark"},
    {"id": "wilder1978", "paper": "Wilder (1978)", "topic": "RSI, ATR, ADX, Supertrend construction"},
    {"id": "kalman1960", "paper": "Kalman (1960)", "topic": "1-D level/trend filter on log-price"},
    {"id": "cmf", "paper": "Chaikin; Granville OBV", "topic": "Money-flow / accumulation"},
    {"id": "ichimoku", "paper": "Hosoda (Ichimoku Kinko Hyo)", "topic": "Cloud trend equilibrium"},
    {"id": "heston2008", "paper": "Heston, Korajczyk, Sadka (intraday)", "topic": "Time-of-day volume/return seasonality"},
    {"id": "crabel1990", "paper": "Crabel (1990) Opening Range", "topic": "Opening-range breakout (intraday app)"},
]


def _arr(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _last(s: pd.Series) -> Optional[float]:
    if s is None or len(s) == 0:
        return None
    v = s.iloc[-1]
    if pd.isna(v) or not np.isfinite(v):
        return None
    return float(v)


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


def log_returns(close: pd.Series) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    return np.log(c / c.shift(1))


def _linreg(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 12:
        return None, None, None
    xm, ym = float(x.mean()), float(y.mean())
    varx = float(((x - xm) ** 2).sum())
    if varx < 1e-18:
        return None, None, None
    beta = float(((x - xm) * (y - ym)).sum() / varx)
    alpha = ym - beta * xm
    resid = y - (alpha + beta * x)
    dof = max(len(x) - 2, 1)
    sigma = float(np.sqrt((resid ** 2).sum() / dof))
    se_b = sigma / np.sqrt(varx) if varx > 0 else np.nan
    tstat = beta / se_b if se_b and np.isfinite(se_b) and se_b > 0 else None
    return alpha, beta, tstat


def max_drawdown(close: pd.Series) -> float:
    c = pd.to_numeric(close, errors="coerce").dropna()
    if c.empty:
        return 0.0
    peak = c.cummax()
    dd = c / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def ulcer_index(close: pd.Series, window: Optional[int] = None) -> Optional[float]:
    c = pd.to_numeric(close, errors="coerce").dropna()
    if len(c) < 8:
        return None
    if window:
        c = c.iloc[-int(window):]
    peak = c.cummax()
    dd_pct = 100.0 * (c / peak - 1.0)
    return float(np.sqrt((dd_pct ** 2).mean()))


def sharpe(r: pd.Series, ann: int) -> Optional[float]:
    x = r.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 8 or float(x.std()) < 1e-12:
        return None
    return float(np.sqrt(ann) * x.mean() / x.std())


def sortino(r: pd.Series, ann: int) -> Optional[float]:
    x = r.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 8:
        return None
    down = x[x < 0]
    dstd = float(down.std()) if len(down) > 2 else float(x.std())
    if dstd < 1e-12:
        return None
    return float(np.sqrt(ann) * x.mean() / dstd)


def calmar(r: pd.Series, close: pd.Series, ann: int) -> Optional[float]:
    x = r.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 8:
        return None
    mdd = abs(max_drawdown(close))
    if mdd < 1e-8:
        return None
    return float(x.mean() * ann / mdd)


def hist_var_cvar(r: pd.Series, q: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    x = r.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 20:
        return None, None
    var = float(np.quantile(x, q))
    tail = x[x <= var]
    cvar = float(tail.mean()) if len(tail) else var
    return var, cvar


def parkinson_vol(high: pd.Series, low: pd.Series, ann: int, window: int = 20) -> Optional[float]:
    h, l = pd.to_numeric(high, errors="coerce"), pd.to_numeric(low, errors="coerce")
    rs = np.log((h / l).replace(0, np.nan)) ** 2
    m = rs.rolling(window, min_periods=max(5, window // 2)).mean()
    v = _last(m)
    if v is None or v <= 0:
        return None
    return float(np.sqrt(ann * v / (4.0 * np.log(2.0))))


def garman_klass_vol(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                     ann: int, window: int = 20) -> Optional[float]:
    o = pd.to_numeric(open_, errors="coerce")
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    hl = np.log((h / l).replace(0, np.nan)) ** 2
    co = np.log((c / o).replace(0, np.nan)) ** 2
    gk = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
    m = gk.rolling(window, min_periods=max(5, window // 2)).mean()
    v = _last(m)
    if v is None or v <= 0:
        return None
    return float(np.sqrt(ann * v))


def yang_zhang_vol(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                   ann: int, window: int = 20) -> Optional[float]:
    o = pd.to_numeric(open_, errors="coerce")
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    n = max(5, min(window, len(c) - 2))
    log_ho = np.log((h / o).replace(0, np.nan))
    log_lo = np.log((l / o).replace(0, np.nan))
    log_co = np.log((c / o).replace(0, np.nan))
    log_oc = np.log((o / c.shift(1)).replace(0, np.nan))
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    overnight = log_oc.rolling(n, min_periods=max(4, n // 2)).var()
    oc_var = log_co.rolling(n, min_periods=max(4, n // 2)).var()
    rs_mean = rs.rolling(n, min_periods=max(4, n // 2)).mean()
    yz = overnight + k * oc_var + (1.0 - k) * rs_mean
    v = _last(yz)
    if v is None or v <= 0:
        return None
    return float(np.sqrt(ann * v))


def ewma_vol(r: pd.Series, ann: int, lam: float = 0.94) -> Optional[float]:
    x = r.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(x) < 15:
        return None
    var = float(np.var(x[:20])) if len(x) >= 20 else float(np.var(x))
    for e in x:
        var = lam * var + (1.0 - lam) * e * e
    return float(np.sqrt(max(var, 0.0) * ann))


def garch11_vol(r: pd.Series, ann: int) -> Optional[float]:
    """Variance-targeting GARCH(1,1) with typical equity params (ω, α=0.06, β=0.92)."""
    x = r.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(x) < 30:
        return None
    alpha, beta = 0.06, 0.92
    omega = max(float(np.var(x)) * (1.0 - alpha - beta), 1e-12)
    var = float(np.var(x))
    for e in x:
        var = omega + alpha * e * e + beta * var
    return float(np.sqrt(max(var, 0.0) * ann))


def hurst_rs(close: pd.Series, max_lag: int = 64) -> Optional[float]:
    """Rescaled-range Hurst. ~0.5 random walk, >0.5 trend, <0.5 mean-reverting."""
    c = pd.to_numeric(close, errors="coerce").dropna()
    r = np.diff(np.log(c.to_numpy(dtype=float) + 1e-12))
    n = len(r)
    if n < 32:
        return None
    lags = [2 ** i for i in range(3, int(np.log2(min(max_lag, n // 2))) + 1)]
    if len(lags) < 3:
        return None
    rs_vals = []
    for lag in lags:
        chunks = n // lag
        if chunks < 2:
            continue
        vals = []
        for i in range(chunks):
            seg = r[i * lag:(i + 1) * lag]
            z = np.cumsum(seg - seg.mean())
            rng = z.max() - z.min()
            s = seg.std()
            if s > 1e-12:
                vals.append(rng / s)
        if vals:
            rs_vals.append((np.log(lag), np.log(np.mean(vals))))
    if len(rs_vals) < 3:
        return None
    xs = np.array([p[0] for p in rs_vals])
    ys = np.array([p[1] for p in rs_vals])
    slope, _ = np.polyfit(xs, ys, 1)
    return float(np.clip(slope, 0.0, 1.0))


def variance_ratio(r: pd.Series, q: int = 4) -> Optional[float]:
    """Lo–MacKinlay VR(q). 1 ≈ random walk, >1 momentum, <1 reversal."""
    x = r.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < q * 8:
        return None
    mu = x.mean()
    var1 = np.sum((x - mu) ** 2) / (n - 1)
    xq = np.array([np.sum(x[i:i + q]) for i in range(0, n - q + 1, 1)])
    varq = np.sum((xq - q * mu) ** 2) / (len(xq) * q)
    if var1 < 1e-18:
        return None
    return float(varq / var1)


def adf_tau(close: pd.Series) -> Optional[float]:
    """Dickey–Fuller τ on log-price (no constant/trend). More negative → more stationary."""
    c = pd.to_numeric(close, errors="coerce").dropna()
    if len(c) < 24:
        return None
    y = np.log(c.to_numpy(dtype=float) + 1e-12)
    dy = np.diff(y)
    ylag = y[:-1]
    a, b, tstat = _linreg(ylag, dy)
    return float(tstat) if tstat is not None else None


def ou_halflife(close: pd.Series) -> Optional[float]:
    """AR(1) half-life in bars of demeaned log-price (Ornstein–Uhlenbeck)."""
    c = pd.to_numeric(close, errors="coerce").dropna()
    if len(c) < 24:
        return None
    y = np.log(c.to_numpy(dtype=float) + 1e-12)
    y = y - y.mean()
    ylag, ynow = y[:-1], y[1:]
    a, b, _ = _linreg(ylag, ynow)
    if b is None or b <= 0 or b >= 1:
        return None
    hl = -np.log(2.0) / np.log(b)
    if not np.isfinite(hl) or hl <= 0 or hl > 5000:
        return None
    return float(hl)


def amihud_illiquidity(r: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> Optional[float]:
    dv = pd.to_numeric(close, errors="coerce") * pd.to_numeric(volume, errors="coerce")
    illiq = r.abs() / (dv.replace(0, np.nan))
    m = illiq.rolling(window, min_periods=max(5, window // 2)).mean()
    v = _last(m)
    if v is None:
        return None
    return float(v * 1e6)


def efficiency_ratio(close: pd.Series, n: int = 20) -> Optional[float]:
    c = pd.to_numeric(close, errors="coerce")
    if len(c) < n + 2:
        return None
    change = abs(float(c.iloc[-1] - c.iloc[-n - 1]))
    path = float(c.diff().abs().iloc[-n:].sum())
    if path < 1e-12:
        return None
    return float(change / path)


def kalman_level(close: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """Scalar Kalman on log-price; returns (filtered last price, slope of last 5 filtered)."""
    c = pd.to_numeric(close, errors="coerce").dropna().to_numpy(dtype=float)
    if len(c) < 10:
        return None, None
    y = np.log(c + 1e-12)
    x, p, q, r = y[0], 1.0, 1e-4, 1e-3
    filt = []
    for obs in y:
        p = p + q
        k = p / (p + r)
        x = x + k * (obs - x)
        p = (1.0 - k) * p
        filt.append(x)
    px = float(np.exp(filt[-1]))
    if len(filt) >= 6:
        slope = float(filt[-1] - filt[-6])
    else:
        slope = 0.0
    return px, slope


def jump_share(r: pd.Series, window: int = 40) -> Optional[float]:
    """Share of recent |r| exceeding 3σ — BNS-style jump intensity proxy."""
    x = r.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < window:
        return None
    rec = x.iloc[-window:]
    sig = float(rec.std())
    if sig < 1e-12:
        return 0.0
    return float((rec.abs() > 3.0 * sig).mean())


def vol_regime(r: pd.Series, window: int = 20) -> str:
    x = r.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < window + 5:
        return "unknown"
    rv = x.rolling(window).std()
    last = float(rv.iloc[-1]) if pd.notna(rv.iloc[-1]) else None
    med = float(rv.median())
    if last is None or med < 1e-12:
        return "unknown"
    if last > med * 1.35:
        return "high_vol"
    if last < med * 0.75:
        return "low_vol"
    return "normal"


def kelly_fraction(r: pd.Series) -> Optional[float]:
    x = r.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 20:
        return None
    mu, var = float(x.mean()), float(x.var())
    if var < 1e-16:
        return None
    f = mu / var
    return float(np.clip(0.5 * f, -0.5, 0.5))  # half-Kelly, capped


def momentum_skip(close: pd.Series, long_bars: int, skip: int) -> Optional[float]:
    """Jegadeesh–Titman style: return from t-long to t-skip (skip most recent bar(s))."""
    c = pd.to_numeric(close, errors="coerce").dropna()
    need = long_bars + skip + 1
    if len(c) < need:
        long_bars = min(long_bars, max(3, len(c) - skip - 2))
        if long_bars < 3:
            return None
    end = -skip - 1 if skip > 0 else -1
    start = end - long_bars
    try:
        a, b = float(c.iloc[start]), float(c.iloc[end])
        if a <= 0:
            return None
        return b / a - 1.0
    except Exception:
        return None


def reversal(close: pd.Series, bars: int) -> Optional[float]:
    c = pd.to_numeric(close, errors="coerce").dropna()
    if len(c) < bars + 2:
        return None
    a, b = float(c.iloc[-bars - 1]), float(c.iloc[-1])
    if a <= 0:
        return None
    return b / a - 1.0


def relative_strength(close: pd.Series, bench: Optional[pd.Series], bars: int) -> Optional[float]:
    if bench is None or bench.empty:
        return None
    a = pd.to_numeric(close, errors="coerce")
    b = pd.to_numeric(bench, errors="coerce")
    joined = pd.concat([a, b], axis=1, join="inner")
    joined.columns = ["a", "b"]
    if len(joined) < bars + 2:
        return None
    ra = float(joined["a"].iloc[-1] / joined["a"].iloc[-bars - 1] - 1.0)
    rb = float(joined["b"].iloc[-1] / joined["b"].iloc[-bars - 1] - 1.0)
    return ra - rb


def ichimoku_signal(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> Optional[float]:
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    conv_n = max(3, n // 3)
    base_n = max(conv_n + 1, (2 * n) // 3)
    tenkan = (h.rolling(conv_n).max() + l.rolling(conv_n).min()) / 2.0
    kijun = (h.rolling(base_n).max() + l.rolling(base_n).min()) / 2.0
    span_a = ((tenkan + kijun) / 2.0).shift(base_n)
    span_b = ((h.rolling(n).max() + l.rolling(n).min()) / 2.0).shift(base_n)
    px, sa, sb, kj = _last(c), _last(span_a), _last(span_b), _last(kijun)
    tk = _last(tenkan)
    if None in (px, sa, sb, kj, tk):
        return None
    cloud_top, cloud_bot = max(sa, sb), min(sa, sb)
    score = 0.0
    if px > cloud_top:
        score += 0.5
    elif px < cloud_bot:
        score -= 0.5
    if tk > kj:
        score += 0.3
    else:
        score -= 0.3
    if px > kj:
        score += 0.2
    else:
        score -= 0.2
    return _clip(score)


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, mult: float = 3.0) -> Optional[float]:
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    atr = _wilder_atr(h, l, c, period)
    hl2 = (h + l) / 2.0
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    st = pd.Series(index=c.index, dtype=float)
    dirn = pd.Series(1, index=c.index)
    for i in range(1, len(c)):
        if c.iloc[i] > upper.iloc[i - 1]:
            dirn.iloc[i] = 1
        elif c.iloc[i] < lower.iloc[i - 1]:
            dirn.iloc[i] = -1
        else:
            dirn.iloc[i] = dirn.iloc[i - 1]
            if dirn.iloc[i] == 1:
                lower.iloc[i] = max(lower.iloc[i], lower.iloc[i - 1])
            else:
                upper.iloc[i] = min(upper.iloc[i], upper.iloc[i - 1])
        st.iloc[i] = lower.iloc[i] if dirn.iloc[i] == 1 else upper.iloc[i]
    d = int(dirn.iloc[-1]) if len(dirn) else 0
    return float(d)


def keltner_position(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> Optional[float]:
    c = pd.to_numeric(close, errors="coerce")
    ema = c.ewm(span=n, adjust=False).mean()
    atr = _wilder_atr(pd.to_numeric(high, errors="coerce"), pd.to_numeric(low, errors="coerce"), c, n)
    upper, lower = ema + 2.0 * atr, ema - 2.0 * atr
    px, u, m, lo = _last(c), _last(upper), _last(ema), _last(lower)
    if None in (px, u, m, lo) or u <= lo:
        return None
    return _clip(2.0 * (px - m) / (u - lo))


def donchian_position(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> Optional[float]:
    h = pd.to_numeric(high, errors="coerce").rolling(n).max()
    l = pd.to_numeric(low, errors="coerce").rolling(n).min()
    c = pd.to_numeric(close, errors="coerce")
    px, hh, ll = _last(c), _last(h), _last(l)
    if None in (px, hh, ll) or hh <= ll:
        return None
    return _clip(2.0 * (px - ll) / (hh - ll) - 1.0)


def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 20) -> Optional[float]:
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    v = pd.to_numeric(volume, errors="coerce")
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    mfv = mfm * v
    cmf = mfv.rolling(n).sum() / v.rolling(n).sum().replace(0, np.nan)
    return _last(cmf)


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14) -> Optional[float]:
    tp = (pd.to_numeric(high) + pd.to_numeric(low) + pd.to_numeric(close)) / 3.0
    mf = tp * pd.to_numeric(volume, errors="coerce")
    delta = tp.diff()
    pos = mf.where(delta > 0, 0.0).rolling(n).sum()
    neg = mf.where(delta < 0, 0.0).rolling(n).sum()
    ratio = pos / (neg.abs() + 1e-12)
    val = 100.0 - (100.0 / (1.0 + ratio))
    return _last(val)


def vwap_series(df: pd.DataFrame) -> pd.Series:
    c = pd.to_numeric(df["close"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    typical = c
    if "high" in df.columns and "low" in df.columns:
        typical = (pd.to_numeric(df["high"]) + pd.to_numeric(df["low"]) + c) / 3.0
    cv = (typical * v).cumsum()
    vs = v.cumsum().replace(0, np.nan)
    return cv / vs


def heikin_ashi_bias(df: pd.DataFrame, n: int = 5) -> Optional[float]:
    if "open" not in df.columns:
        return None
    o = pd.to_numeric(df["open"], errors="coerce")
    h = pd.to_numeric(df.get("high", df["close"]), errors="coerce")
    l = pd.to_numeric(df.get("low", df["close"]), errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")
    ha_c = (o + h + l + c) / 4.0
    ha_o = (o + c) / 2.0
    ha_o = ha_o.copy()
    for i in range(1, len(ha_o)):
        ha_o.iloc[i] = (ha_o.iloc[i - 1] + ha_c.iloc[i - 1]) / 2.0
    body = ha_c - ha_o
    last = body.iloc[-n:]
    if last.isna().all():
        return None
    return _clip(float(np.sign(last).mean()))


def _mom_lookback(ann: int, n: int) -> Tuple[int, int, int]:
    """(long, skip, reversal_window) in bars."""
    if ann >= 200:  # daily
        long_b, skip, rev = 252, 21, 21
    elif ann >= 10:  # monthly
        long_b, skip, rev = 12, 1, 1
    elif ann >= 3:  # quarterly
        long_b, skip, rev = 8, 1, 1
    else:
        long_b, skip, rev = 6, 1, 1
    long_b = min(long_b, max(3, n - skip - 3))
    return long_b, skip, min(rev, max(1, n // 8))


def quant_signal_score(bundle: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Map research features into a bounded directional score and reasons."""
    reasons: List[str] = []
    score = 0.0
    n = 0

    def add(val: Optional[float], w: float, why: str, bull_if_pos: bool = True):
        nonlocal score, n
        if val is None or not np.isfinite(val):
            return
        signed = val if bull_if_pos else -val
        score += _clip(signed) * w
        n += 1
        reasons.append(why)

    mom = bundle.get("momentum_skip_pct")
    if mom is not None:
        add(np.clip(mom / 25.0, -1, 1), 0.35, f"JT momentum {mom:+.1f}%")

    rev = bundle.get("reversal_pct")
    hurst = bundle.get("hurst")
    if rev is not None and hurst is not None and hurst < 0.45:
        add(-np.clip(rev / 8.0, -1, 1), 0.25, f"Mean-reversion fade {rev:+.1f}% (H={hurst:.2f})")
    elif rev is not None and hurst is not None and hurst > 0.55:
        add(np.clip(rev / 8.0, -1, 1), 0.15, f"Trend continuation {rev:+.1f}% (H={hurst:.2f})")

    rs = bundle.get("rel_strength_pct")
    if rs is not None:
        add(np.clip(rs / 12.0, -1, 1), 0.2, f"vs benchmark {rs:+.1f}%")

    beta = bundle.get("beta")
    if beta is not None and beta > 1.4:
        add(-0.25, 0.15, f"High beta {beta:.2f} (BAB / AHXZ)")
    elif beta is not None and 0 < beta < 0.7:
        add(0.2, 0.12, f"Low beta {beta:.2f} (BAB)")

    ich = bundle.get("ichimoku")
    if ich is not None:
        add(ich, 0.2, "Ichimoku cloud " + ("bull" if ich > 0 else "bear"))

    st = bundle.get("supertrend")
    if st is not None:
        add(st, 0.18, "Supertrend " + ("long" if st > 0 else "short"))

    ha = bundle.get("heikin_ashi")
    if ha is not None:
        add(ha, 0.1, "Heikin-Ashi bias")

    cmf = bundle.get("cmf")
    if cmf is not None:
        add(np.clip(cmf * 3, -1, 1), 0.12, f"Chaikin MF {cmf:+.2f}")

    er = bundle.get("efficiency_ratio")
    adx = bundle.get("adx_proxy")
    if er is not None and er < 0.25:
        reasons.append(f"Choppy (Kaufman ER {er:.2f}) — fade extremes")
    elif er is not None and er > 0.45:
        reasons.append(f"Efficient trend (ER {er:.2f})")

    kalman_slope = bundle.get("kalman_slope")
    if kalman_slope is not None:
        add(np.clip(kalman_slope * 8, -1, 1), 0.12, "Kalman trend " + ("up" if kalman_slope > 0 else "down"))

    sharpe_v = bundle.get("sharpe")
    if sharpe_v is not None:
        add(np.clip(sharpe_v / 2.0, -1, 1), 0.1, f"Sharpe {sharpe_v:.2f}")

    if n == 0:
        return 0.0, reasons
    return _clip(score / max(0.6, 0.35 * n + 0.2)), reasons


def compute_quant_bundle(
    df: pd.DataFrame,
    ticker: str,
    candle: str = "24h",
    benchmark: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    from ohlc import lookbacks, normalize_candle
    from markets import currency_for, market_for, benchmark_symbol

    cndl = normalize_candle(candle)
    if df is None or df.empty or "close" not in df.columns:
        return {"error": "no data", "ticker": ticker, "candle": cndl}

    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce") if "high" in df.columns else close
    low = pd.to_numeric(df["low"], errors="coerce") if "low" in df.columns else close
    open_ = pd.to_numeric(df["open"], errors="coerce") if "open" in df.columns else close
    volume = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else pd.Series(0.0, index=df.index)

    lb = lookbacks(cndl, len(close))
    ann = int(lb["ann"])
    r = log_returns(close)

    long_b, skip, rev_b = _mom_lookback(ann, len(close))
    mom = momentum_skip(close, long_b, skip)
    rev = reversal(close, rev_b)
    rs = relative_strength(close, benchmark, max(rev_b, min(long_b, len(close) // 3)))

    alpha = beta = beta_t = None
    if benchmark is not None and not benchmark.empty:
        joined = pd.concat([r.rename("y"), log_returns(benchmark).rename("x")], axis=1, join="inner").dropna()
        if len(joined) >= 15:
            a, b, t = _linreg(joined["x"].to_numpy(), joined["y"].to_numpy())
            if a is not None:
                alpha = float(a * ann)
                beta = float(b)
                beta_t = t

    var95, cvar95 = hist_var_cvar(r, 0.05)
    sh = sharpe(r, ann)
    so = sortino(r, ann)
    cal = calmar(r, close, ann)
    mdd = max_drawdown(close)
    ui = ulcer_index(close)
    pk = parkinson_vol(high, low, ann, lb["vol"])
    gk = garman_klass_vol(open_, high, low, close, ann, lb["vol"])
    yz = yang_zhang_vol(open_, high, low, close, ann, lb["vol"])
    ew = ewma_vol(r, ann)
    garch = garch11_vol(r, ann)
    hst = hurst_rs(close)
    vr = variance_ratio(r, q=4 if len(r) > 40 else 2)
    tau = adf_tau(close)
    hl = ou_halflife(close)
    ami = amihud_illiquidity(r, close, volume, lb["vol"])
    er = efficiency_ratio(close, lb["er"])
    kpx, kslope = kalman_level(close)
    jumps = jump_share(r, max(20, lb["vol"] * 2))
    regime = vol_regime(r, lb["vol"])
    kelly = kelly_fraction(r)
    ich = ichimoku_signal(high, low, close, max(12, lb["sma_l"]))
    try:
        st = supertrend(high, low, close, max(7, lb["atr"]), 3.0)
    except Exception:
        st = None
    kel = keltner_position(high, low, close, lb["bb"])
    don = donchian_position(high, low, close, lb["bb"])
    cmf = chaikin_money_flow(high, low, close, volume, lb["vol"])
    mfi_v = mfi(high, low, close, volume, lb["rsi"])
    ha = heikin_ashi_bias(df)
    vw = vwap_series(df)
    vwap_dev = None
    if _last(vw) and float(close.iloc[-1]) > 0:
        vwap_dev = float(close.iloc[-1]) / float(_last(vw)) - 1.0

    skew = float(r.dropna().skew()) if len(r.dropna()) > 12 else None
    kurt = float(r.dropna().kurt()) if len(r.dropna()) > 12 else None

    idio = None
    if beta is not None and benchmark is not None:
        joined = pd.concat([r.rename("y"), log_returns(benchmark).rename("x")], axis=1, join="inner").dropna()
        if len(joined) >= 20:
            resid = joined["y"] - (beta * joined["x"])
            idio = float(resid.std() * np.sqrt(ann))

    bab = None
    if beta is not None and sh is not None and beta != 0:
        bab = float(sh / max(abs(beta), 0.2))

    bundle: Dict[str, Any] = {
        "ticker": ticker,
        "candle": cndl,
        "market": market_for(ticker),
        "currency": currency_for(ticker),
        "benchmark": benchmark_symbol(ticker),
        "bars": int(len(close)),
        "price": float(close.iloc[-1]),
        "sharpe": round(sh, 3) if sh is not None else None,
        "sortino": round(so, 3) if so is not None else None,
        "calmar": round(cal, 3) if cal is not None else None,
        "max_drawdown_pct": round(mdd * 100, 2),
        "ulcer": round(ui, 3) if ui is not None else None,
        "var95_pct": round(var95 * 100, 3) if var95 is not None else None,
        "cvar95_pct": round(cvar95 * 100, 3) if cvar95 is not None else None,
        "beta": round(beta, 3) if beta is not None else None,
        "alpha_ann_pct": round(alpha * 100, 2) if alpha is not None else None,
        "beta_tstat": round(beta_t, 2) if beta_t is not None else None,
        "idio_vol_ann": round(idio, 3) if idio is not None else None,
        "bab_score": round(bab, 3) if bab is not None else None,
        "vol_parkinson": round(pk, 3) if pk is not None else None,
        "vol_garman_klass": round(gk, 3) if gk is not None else None,
        "vol_yang_zhang": round(yz, 3) if yz is not None else None,
        "vol_ewma": round(ew, 3) if ew is not None else None,
        "vol_garch11": round(garch, 3) if garch is not None else None,
        "hurst": round(hst, 3) if hst is not None else None,
        "variance_ratio": round(vr, 3) if vr is not None else None,
        "adf_tau": round(tau, 2) if tau is not None else None,
        "ou_halflife_bars": round(hl, 1) if hl is not None else None,
        "amihud": round(ami, 6) if ami is not None else None,
        "efficiency_ratio": round(er, 3) if er is not None else None,
        "kalman_price": round(kpx, 4) if kpx is not None else None,
        "kalman_slope": round(kslope, 5) if kslope is not None else None,
        "jump_share": round(jumps, 3) if jumps is not None else None,
        "vol_regime": regime,
        "kelly_half": round(kelly, 4) if kelly is not None else None,
        "momentum_skip_pct": round(mom * 100, 2) if mom is not None else None,
        "reversal_pct": round(rev * 100, 2) if rev is not None else None,
        "rel_strength_pct": round(rs * 100, 2) if rs is not None else None,
        "ichimoku": round(ich, 3) if ich is not None else None,
        "supertrend": st,
        "keltner_pos": round(kel, 3) if kel is not None else None,
        "donchian_pos": round(don, 3) if don is not None else None,
        "cmf": round(cmf, 3) if cmf is not None else None,
        "mfi": round(mfi_v, 1) if mfi_v is not None else None,
        "heikin_ashi": round(ha, 3) if ha is not None else None,
        "vwap_dev_pct": round(vwap_dev * 100, 2) if vwap_dev is not None else None,
        "skew": round(skew, 3) if skew is not None else None,
        "excess_kurtosis": round(kurt, 3) if kurt is not None else None,
        "lookbacks": lb,
    }
    qscore, qreasons = quant_signal_score(bundle)
    bundle["quant_score"] = round(qscore, 3)
    bundle["quant_reasons"] = qreasons
    bundle["hurst_read"] = (
        "trending" if hst is not None and hst > 0.55
        else "mean_reverting" if hst is not None and hst < 0.45
        else "random_walk"
    )
    return bundle


def try_fundamentals(ticker: str) -> Dict[str, Any]:
    """Best-effort yfinance fundamentals (slow; research view only)."""
    import yfinance as yf

    out: Dict[str, Any] = {"available": False}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        out["error"] = str(e)
        return out
    keys = [
        "trailingPE", "forwardPE", "priceToBook", "profitMargins", "returnOnEquity",
        "returnOnAssets", "debtToEquity", "currentRatio", "freeCashflow",
        "operatingCashflow", "revenueGrowth", "earningsGrowth", "pegRatio",
        "enterpriseToEbitda", "shortPercentOfFloat", "beta", "marketCap",
        "dividendYield", "payoutRatio", "bookValue", "enterpriseValue",
    ]
    snap = {}
    for k in keys:
        v = info.get(k)
        if isinstance(v, (int, float)) and np.isfinite(v):
            snap[k] = round(float(v), 4)
        elif v is not None and not isinstance(v, (dict, list)):
            snap[k] = v
    if not snap:
        return out
    # Crude Piotroski-like points from available fields (not the full 9-signal F-score)
    f_proxy = 0
    if snap.get("returnOnEquity") and snap["returnOnEquity"] > 0:
        f_proxy += 1
    if snap.get("returnOnAssets") and snap["returnOnAssets"] > 0:
        f_proxy += 1
    if snap.get("operatingCashflow") and snap["operatingCashflow"] > 0:
        f_proxy += 1
    if snap.get("currentRatio") and snap["currentRatio"] > 1:
        f_proxy += 1
    if snap.get("debtToEquity") is not None and snap["debtToEquity"] < 100:
        f_proxy += 1
    if snap.get("revenueGrowth") and snap["revenueGrowth"] > 0:
        f_proxy += 1
    if snap.get("profitMargins") and snap["profitMargins"] > 0:
        f_proxy += 1
    out.update({"available": True, "snapshot": snap, "quality_points": f_proxy, "quality_points_max": 7})
    return out


def cross_section_ranks(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cross-sectional ranks (AQR-style) within the scanned batch."""
    if len(rows) < 4:
        return rows

    def rank_key(key: str, higher_better: bool = True) -> Dict[str, float]:
        vals = []
        for r in rows:
            q = r.get("quant") or {}
            v = q.get(key)
            if v is not None:
                try:
                    vals.append((r["ticker"], float(v)))
                except (TypeError, ValueError):
                    pass
        if len(vals) < 4:
            return {}
        vals.sort(key=lambda x: x[1], reverse=higher_better)
        n = len(vals)
        return {t: round((n - i) / n, 3) for i, (t, _) in enumerate(vals)}

    mom = rank_key("momentum_skip_pct", True)
    lowvol = rank_key("vol_yang_zhang", False)
    sharpe_r = rank_key("sharpe", True)
    rs = rank_key("rel_strength_pct", True)
    for r in rows:
        t = r.get("ticker")
        q = r.setdefault("quant", {})
        q["cs_momentum_rank"] = mom.get(t)
        q["cs_lowvol_rank"] = lowvol.get(t)
        q["cs_sharpe_rank"] = sharpe_r.get(t)
        q["cs_rel_strength_rank"] = rs.get(t)
        ranks = [v for v in (mom.get(t), lowvol.get(t), sharpe_r.get(t), rs.get(t)) if v is not None]
        q["cs_composite"] = round(float(np.mean(ranks)), 3) if ranks else None
    return rows
