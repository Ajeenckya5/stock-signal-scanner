"""
5-minute US + India session microstructure and short-horizon quant.

Yahoo Finance supplies ~60 calendar days of 5m bars. Session clocks:
  NSE  09:15–15:30 Asia/Kolkata
  US   09:30–16:00 America/New_York
"""

from __future__ import annotations

from datetime import time as dtime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ohlc import fetch_ohlcv, lookbacks, to_candles
from markets import benchmark_symbol, currency_for, market_for
from quant_engine import (
    compute_quant_bundle,
    efficiency_ratio,
    hurst_rs,
    kalman_level,
    vwap_series,
)
from scanner import (
    compute_atr,
    compute_bollinger,
    compute_macd,
    compute_momentum,
    compute_rsi,
    compute_sma_crossover,
)

_TZ = {
    "IN": "Asia/Kolkata",
    "US": "America/New_York",
}
_OPEN = {
    "IN": dtime(9, 15),
    "US": dtime(9, 30),
}
_CLOSE = {
    "IN": dtime(15, 30),
    "US": dtime(16, 0),
}


def _intra_forecast(price, action, score, stop, take, support, resistance, atr_v, reasons):
    from forecast import build_forecast
    why = reasons[0] if reasons else None
    return build_forecast(
        price=price,
        action=action,
        score=score,
        stop_loss=stop,
        take_profit=take,
        support=support,
        resistance=resistance,
        atr=atr_v,
        target_days=None,
        ml=None,
        hist=None,
        candle="5m",
        why=why,
    )


def _localize_index(df: pd.DataFrame, mkt: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    tz = _TZ.get(mkt, "UTC")
    idx = out.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    try:
        out.index = idx.tz_convert(tz)
    except Exception:
        out.index = idx
    return out


def session_mask(index: pd.DatetimeIndex, mkt: str) -> pd.Series:
    open_t, close_t = _OPEN[mkt], _CLOSE[mkt]
    t = index.time
    return pd.Series([(open_t <= x <= close_t) for x in t], index=index)


def last_session_frame(df: pd.DataFrame, mkt: str) -> pd.DataFrame:
    if df.empty:
        return df
    days = pd.Series(df.index.date, index=df.index)
    last = days.iloc[-1]
    return df[days == last]


def opening_range(sess: pd.DataFrame, minutes: int = 30) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if sess.empty or "high" not in sess.columns:
        return None, None, None
    start = sess.index[0]
    end = start + pd.Timedelta(minutes=minutes)
    window = sess[(sess.index >= start) & (sess.index < end)]
    if window.empty:
        window = sess.iloc[:6]
    if window.empty:
        return None, None, None
    orh = float(window["high"].max())
    orl = float(window["low"].min())
    mid = (orh + orl) / 2.0
    return orh, orl, mid


def volume_profile(sess: pd.DataFrame, bins: int = 24) -> Dict[str, Any]:
    if sess.empty or "close" not in sess.columns:
        return {"bins": [], "poc": None, "vah": None, "val": None}
    px = pd.to_numeric(sess["close"], errors="coerce")
    vol = pd.to_numeric(sess.get("volume", 0.0), errors="coerce").fillna(0.0)
    lo, hi = float(px.min()), float(px.max())
    if hi <= lo:
        return {"bins": [], "poc": float(px.iloc[-1]), "vah": None, "val": None}
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(px.to_numpy(), edges) - 1, 0, bins - 1)
    vol_bins = np.zeros(bins)
    for i, v in zip(idx, vol.to_numpy()):
        vol_bins[i] += float(v)
    centers = 0.5 * (edges[:-1] + edges[1:])
    poc_i = int(np.argmax(vol_bins))
    poc = float(centers[poc_i])
    total = vol_bins.sum()
    vah = val = poc
    if total > 0:
        order = np.argsort(vol_bins)[::-1]
        acc = 0.0
        kept = []
        for i in order:
            acc += vol_bins[i]
            kept.append(i)
            if acc >= 0.7 * total:
                break
        vah = float(centers[max(kept)])
        val = float(centers[min(kept)])
    bins_out = [
        {"price": round(float(centers[i]), 4), "volume": round(float(vol_bins[i]), 2)}
        for i in range(bins)
        if vol_bins[i] > 0
    ]
    return {"bins": bins_out, "poc": round(poc, 4), "vah": round(vah, 4), "val": round(val, 4)}


def relative_volume_tod(df: pd.DataFrame, sess: pd.DataFrame) -> Optional[float]:
    """Current bar volume vs median volume at the same clock time across days."""
    if df.empty or sess.empty or "volume" not in df.columns:
        return None
    last_t = sess.index[-1].time()
    same = df[df.index.map(lambda x: x.time() == last_t)]
    if len(same) < 4:
        return None
    med = float(same["volume"].median())
    cur = float(sess["volume"].iloc[-1])
    if med <= 0:
        return None
    return cur / med


def cumulative_delta(sess: pd.DataFrame) -> Optional[float]:
    """Tick-rule signed volume (close vs prior close) — Lee-Ready proxy without quotes."""
    if sess.empty or len(sess) < 3:
        return None
    c = pd.to_numeric(sess["close"], errors="coerce")
    v = pd.to_numeric(sess["volume"], errors="coerce").fillna(0.0)
    sign = np.sign(c.diff().fillna(0.0))
    sign = sign.replace(0, np.nan).ffill().fillna(1.0)
    return float((sign * v).sum())


def gap_pct(df: pd.DataFrame, sess: pd.DataFrame) -> Optional[float]:
    if df.empty or sess.empty:
        return None
    prev = df[df.index.date < sess.index[0].date()]
    if prev.empty:
        return None
    prior_close = float(prev["close"].iloc[-1])
    sess_open = float(sess["open"].iloc[0]) if "open" in sess.columns else float(sess["close"].iloc[0])
    if prior_close <= 0:
        return None
    return sess_open / prior_close - 1.0


def tod_seasonality(df: pd.DataFrame, sess: pd.DataFrame) -> Optional[float]:
    """Mean historical return of this clock slot (Heston–Korajczyk–Sadka style)."""
    if df.empty or sess.empty or len(df) < 80:
        return None
    t = sess.index[-1].time()
    r = np.log(pd.to_numeric(df["close"], errors="coerce") / pd.to_numeric(df["close"], errors="coerce").shift(1))
    mask = df.index.map(lambda x: x.time() == t)
    sl = r[list(mask)].dropna()
    if len(sl) < 5:
        return None
    return float(sl.mean())


def _binary(score: float) -> str:
    if score >= 0.35:
        return "BUY"
    if score <= -0.35:
        return "SELL"
    return "BUY" if score >= 0 else "SELL"


def analyze_intraday(ticker: str) -> Dict[str, Any]:
    t = str(ticker).strip().upper()
    mkt = market_for(t)
    if mkt == "INDEX":
        mkt = "US"
    df, meta = fetch_ohlcv(t, candle="5m")
    if df.empty:
        return {
            "ticker": t,
            "action": "ERROR",
            "error": meta.get("error") or "No 5-minute data (Yahoo ~60d, regular session).",
            "market": mkt,
            "currency": currency_for(t),
        }

    df = _localize_index(df, mkt)
    mask = session_mask(df.index, mkt)
    rth = df.loc[mask.values] if len(mask) == len(df) else df
    if rth.empty:
        rth = df
    sess = last_session_frame(rth, mkt)
    if sess.empty:
        sess = rth.iloc[-78:] if len(rth) > 78 else rth

    close = pd.to_numeric(rth["close"], errors="coerce")
    high = pd.to_numeric(rth["high"], errors="coerce") if "high" in rth.columns else close
    low = pd.to_numeric(rth["low"], errors="coerce") if "low" in rth.columns else close
    volume = pd.to_numeric(rth["volume"], errors="coerce") if "volume" in rth.columns else pd.Series(1.0, index=rth.index)
    lb = lookbacks("5m", len(close))

    rsi = compute_rsi(close, lb["rsi"])
    macd_line, sig, hist = compute_macd(close)
    bb_u, bb_m, bb_l = compute_bollinger(close, lb["bb"])
    sma_s, sma_l = compute_sma_crossover(close, lb["sma_s"], lb["sma_l"])
    mom = compute_momentum(close, lb["mom"])
    atr = compute_atr(high, low, close, lb["atr"])

    price = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else price
    change_pct = (price / prev - 1.0) * 100 if prev else 0.0
    rsi_v = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None
    macd_h = float(hist.iloc[-1]) if len(hist) and pd.notna(hist.iloc[-1]) else None
    atr_v = float(atr.iloc[-1]) if len(atr) and pd.notna(atr.iloc[-1]) else price * 0.004

    orh, orl, ormid = opening_range(sess, 30)
    vp = volume_profile(sess)
    rvol = relative_volume_tod(rth, sess)
    delta = cumulative_delta(sess)
    gap = gap_pct(rth, sess)
    seas = tod_seasonality(rth, sess)
    vw = vwap_series(sess if len(sess) >= 5 else rth)
    vwap_now = float(vw.iloc[-1]) if len(vw) and pd.notna(vw.iloc[-1]) else None
    vwap_dev = (price / vwap_now - 1.0) if vwap_now else None
    hst = hurst_rs(sess["close"] if len(sess) >= 40 else close)
    er = efficiency_ratio(sess["close"] if len(sess) >= 20 else close, min(20, max(8, len(sess) // 3)))
    _, kslope = kalman_level(sess["close"] if len(sess) >= 15 else close)

    bench_sym = benchmark_symbol(t)
    bench_df, _ = fetch_ohlcv(bench_sym, candle="5m")
    bench_close = None
    if not bench_df.empty:
        bench_df = _localize_index(bench_df, mkt)
        bench_close = pd.to_numeric(bench_df["close"], errors="coerce")
    qbundle = compute_quant_bundle(rth.tail(400), t, candle="5m", benchmark=bench_close)

    reasons: List[str] = []
    score = 0.0

    if orh is not None and price > orh:
        score += 0.45
        reasons.append(f"Opening-range breakout (Crabel) > {orh:.2f}")
    elif orl is not None and price < orl:
        score -= 0.45
        reasons.append(f"Opening-range breakdown < {orl:.2f}")

    if vwap_dev is not None:
        if vwap_dev > 0.002:
            score += 0.25
            reasons.append(f"Above session VWAP ({vwap_dev*100:+.2f}%)")
        elif vwap_dev < -0.002:
            score -= 0.25
            reasons.append(f"Below session VWAP ({vwap_dev*100:+.2f}%)")

    poc = vp.get("poc")
    if poc:
        if price > poc:
            score += 0.12
            reasons.append(f"Above volume POC {poc:.2f}")
        else:
            score -= 0.12
            reasons.append(f"Below volume POC {poc:.2f}")

    if rsi_v is not None:
        if rsi_v < 30:
            score += 0.35
            reasons.append(f"5m RSI oversold ({rsi_v:.0f})")
        elif rsi_v > 70:
            score -= 0.35
            reasons.append(f"5m RSI overbought ({rsi_v:.0f})")

    if macd_h is not None:
        score += float(np.clip(macd_h / (price * 0.002 + 1e-9), -1, 1)) * 0.2
        reasons.append("MACD " + ("bull" if macd_h > 0 else "bear"))

    if rvol is not None and rvol > 1.5:
        if change_pct > 0:
            score += 0.2
            reasons.append(f"Relative volume {rvol:.1f}x on uptick")
        else:
            score -= 0.2
            reasons.append(f"Relative volume {rvol:.1f}x on downtick")

    if delta is not None and sess["volume"].sum() > 0:
        imb = delta / float(sess["volume"].sum())
        score += float(np.clip(imb, -1, 1)) * 0.2
        reasons.append(f"Cumulative delta imbalance {imb:+.2f}")

    if gap is not None and abs(gap) > 0.004:
        # gap-fill mean reversion if Hurst < 0.5
        if hst is not None and hst < 0.5:
            score -= float(np.clip(gap / 0.02, -1, 1)) * 0.15
            reasons.append(f"Gap {gap*100:+.2f}% fade (mean-reverting 5m)")
        else:
            score += float(np.clip(gap / 0.02, -1, 1)) * 0.1
            reasons.append(f"Gap {gap*100:+.2f}% continuation")

    if kslope is not None:
        score += float(np.clip(kslope * 15, -1, 1)) * 0.15
        reasons.append("Kalman 5m " + ("up" if kslope > 0 else "down"))

    if seas is not None:
        score += float(np.clip(seas / 0.0015, -1, 1)) * 0.08

    qs = qbundle.get("quant_score") if isinstance(qbundle, dict) else None
    if qs is not None:
        score += float(qs) * 0.25
        reasons.extend((qbundle.get("quant_reasons") or [])[:3])

    score = float(np.clip(score, -1, 1))
    action = _binary(score)

    if action == "BUY":
        stop = price - 1.4 * atr_v
        take = price + 2.2 * atr_v
        if orl:
            stop = min(stop, orl - 0.1 * atr_v)
    else:
        stop = price + 1.4 * atr_v
        take = price - 2.2 * atr_v
        if orh:
            stop = max(stop, orh + 0.1 * atr_v)

    support = float(sess["low"].min()) if len(sess) else price * 0.99
    resistance = float(sess["high"].max()) if len(sess) else price * 1.01

    vwap_line = []
    if len(vw):
        for idx, val in vw.dropna().items():
            vwap_line.append({"time": int(pd.Timestamp(idx).tz_convert("UTC").timestamp()), "value": float(val)})

    candles = to_candles(sess if len(sess) >= 20 else rth.tail(120), "5m")
    price_lines = []
    if vwap_now:
        price_lines.append({"price": vwap_now, "color": "#eab308", "title": f"VWAP {vwap_now:.2f}"})
    if orh:
        price_lines.append({"price": orh, "color": "#22c55e", "title": f"ORH {orh:.2f}"})
    if orl:
        price_lines.append({"price": orl, "color": "#ef4444", "title": f"ORL {orl:.2f}"})
    if poc:
        price_lines.append({"price": poc, "color": "#38bdf8", "title": f"POC {poc:.2f}"})

    return {
        "ticker": t,
        "market": mkt,
        "currency": currency_for(t),
        "action": action,
        "score": round(score, 3),
        "price": price,
        "change_pct": round(change_pct, 3),
        "rsi": round(rsi_v, 1) if rsi_v is not None else None,
        "macd_hist": round(macd_h, 5) if macd_h is not None else None,
        "atr": round(atr_v, 4),
        "vwap": round(vwap_now, 4) if vwap_now else None,
        "vwap_dev_pct": round(vwap_dev * 100, 3) if vwap_dev is not None else None,
        "orh": round(orh, 4) if orh else None,
        "orl": round(orl, 4) if orl else None,
        "rvol": round(rvol, 2) if rvol is not None else None,
        "cum_delta": round(delta, 0) if delta is not None else None,
        "gap_pct": round(gap * 100, 3) if gap is not None else None,
        "tod_seasonality": round(seas, 6) if seas is not None else None,
        "hurst": round(hst, 3) if hst is not None else None,
        "efficiency_ratio": round(er, 3) if er is not None else None,
        "volume_profile": vp,
        "stop_loss": round(float(stop), 4),
        "take_profit": round(float(take), 4),
        "support": round(support, 4),
        "resistance": round(resistance, 4),
        "reasons": reasons or ["No strong 5m structure"],
        "candle": "5m",
        "forecast": _intra_forecast(price, action, score, stop, take, support, resistance, atr_v, reasons),
        "session_start": str(sess.index[0]) if len(sess) else None,
        "session_end": str(sess.index[-1]) if len(sess) else None,
        "bars_session": int(len(sess)),
        "bars_history": int(len(rth)),
        "quant": {
            "quant_score": qbundle.get("quant_score"),
            "sharpe": qbundle.get("sharpe"),
            "hurst": qbundle.get("hurst"),
            "vol_ewma": qbundle.get("vol_ewma"),
            "kelly_half": qbundle.get("kelly_half"),
            "vol_regime": qbundle.get("vol_regime"),
        },
        "candles": candles,
        "vwap_line": vwap_line,
        "price_lines": price_lines,
        "error": None,
    }


def scan_intraday(tickers: List[str], filter_action: Optional[str] = None) -> List[Dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor

    tickers = [str(x).strip().upper() for x in tickers if str(x).strip()]
    if not tickers:
        return []
    workers = min(6, max(1, len(tickers)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(analyze_intraday, tickers))
    if filter_action:
        rows = [r for r in rows if r.get("action") == filter_action]
    return rows
