"""
OHLCV download + institutional candle construction.

Long-term (this app):
  24h  → daily session bars
  1mo  → monthly bars
  3mo  → calendar-quarter bars (resampled from daily)
  6mo  → semi-annual bars (resampled from daily)

Intraday (companion app):
  5m   → 5-minute bars (Yahoo: last ~60 days)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yfinance as yf

CANDLE_SPECS: Dict[str, Dict[str, Any]] = {
    "24h": {
        "label": "24h / Daily",
        "period": "2y",
        "interval": "1d",
        "resample": None,
        "ann": 252,
        "fwd_bars": 5,
        "min_bars": 40,
        "style": "longterm",
    },
    "1d": {
        "label": "24h / Daily",
        "period": "2y",
        "interval": "1d",
        "resample": None,
        "ann": 252,
        "fwd_bars": 5,
        "min_bars": 40,
        "style": "longterm",
    },
    "1mo": {
        "label": "1 Month",
        "period": "max",
        "interval": "1mo",
        "resample": None,
        "ann": 12,
        "fwd_bars": 1,
        "min_bars": 18,
        "style": "longterm",
    },
    "3mo": {
        "label": "3 Month",
        "period": "max",
        "interval": "1d",
        "resample": "QS",
        "ann": 4,
        "fwd_bars": 1,
        "min_bars": 10,
        "style": "longterm",
    },
    "6mo": {
        "label": "6 Month",
        "period": "max",
        "interval": "1d",
        "resample": "2QS",
        "ann": 2,
        "fwd_bars": 1,
        "min_bars": 8,
        "style": "longterm",
    },
    "5m": {
        "label": "5 Minute",
        "period": "60d",
        "interval": "5m",
        "resample": None,
        "ann": 252 * 78,
        "fwd_bars": 6,
        "min_bars": 50,
        "style": "intraday",
        "prepost": False,
    },
}

LONGTERM_CANDLES = ("24h", "1mo", "3mo", "6mo")


def normalize_candle(candle: Optional[str], default: str = "24h") -> str:
    c = (candle or default).strip().lower()
    aliases = {
        "daily": "24h",
        "day": "24h",
        "1d": "24h",
        "24hr": "24h",
        "24": "24h",
        "month": "1mo",
        "monthly": "1mo",
        "1m": "1mo",
        "1month": "1mo",
        "quarter": "3mo",
        "quarterly": "3mo",
        "3month": "3mo",
        "3m": "3mo",
        "semiannual": "6mo",
        "half": "6mo",
        "6month": "6mo",
        "6m": "6mo",
        "5min": "5m",
        "5minute": "5m",
        "intraday": "5m",
    }
    c = aliases.get(c, c)
    if c not in CANDLE_SPECS:
        return default
    return c


def lookbacks(candle: str, n_bars: int) -> Dict[str, int]:
    """Scale classic 14/20/50 lookbacks to the available bar count."""
    c = normalize_candle(candle)
    spec = CANDLE_SPECS[c]
    n = max(int(n_bars), 1)

    def cap(want: int, floor: int = 3) -> int:
        return max(floor, min(want, max(floor, n // 3)))

    if c in ("24h", "1d"):
        raw = dict(rsi=14, sma_s=10, sma_l=50, mom=10, bb=20, atr=14, stoch=14, adx=14, vol=20, er=20)
    elif c == "1mo":
        raw = dict(rsi=9, sma_s=3, sma_l=12, mom=6, bb=12, atr=8, stoch=9, adx=8, vol=12, er=12)
    elif c == "3mo":
        raw = dict(rsi=6, sma_s=2, sma_l=8, mom=4, bb=8, atr=6, stoch=6, adx=6, vol=8, er=8)
    elif c == "6mo":
        raw = dict(rsi=4, sma_s=2, sma_l=6, mom=3, bb=6, atr=4, stoch=4, adx=4, vol=6, er=6)
    else:  # 5m
        raw = dict(rsi=14, sma_s=9, sma_l=21, mom=12, bb=20, atr=14, stoch=14, adx=14, vol=20, er=20)
    return {k: cap(v) for k, v in raw.items()} | {"ann": int(spec["ann"]), "fwd_bars": int(spec["fwd_bars"])}


def _flatten_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        t = str(ticker)
        sliced = False
        for level in range(out.columns.nlevels):
            vals = [str(v) for v in out.columns.get_level_values(level)]
            if t in vals:
                out = out.xs(t, level=level, axis=1)
                sliced = True
                break
        if not sliced and isinstance(out.columns, pd.MultiIndex):
            out.columns = [
                next((str(x) for x in col if isinstance(x, str) and x), "_".join(str(x) for x in col))
                if isinstance(col, tuple) else str(col)
                for col in out.columns
            ]
    rename = {}
    for c in out.columns:
        lc = str(c).lower().strip()
        if lc in {"open", "high", "low", "close", "volume"}:
            rename[c] = lc
        elif lc in {"adj close", "adjclose"}:
            rename[c] = "close"
    out = out.rename(columns=rename)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    out = out[keep].copy()
    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["close"])
    if "volume" not in out.columns:
        out["volume"] = 0.0
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.isna()]
    if getattr(out.index, "tz", None) is not None:
        try:
            out.index = out.index.tz_convert("UTC")
        except Exception:
            out.index = out.index.tz_localize(None)
    return out.sort_index()


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()
    if getattr(x.index, "tz", None) is not None:
        x.index = x.index.tz_localize(None)
    agg = {}
    if "open" in x.columns:
        agg["open"] = "first"
    if "high" in x.columns:
        agg["high"] = "max"
    if "low" in x.columns:
        agg["low"] = "min"
    agg["close"] = "last"
    if "volume" in x.columns:
        agg["volume"] = "sum"
    out = x.resample(rule).agg(agg).dropna(subset=["close"])
    return out


def fetch_ohlcv(
    ticker: str,
    candle: str = "24h",
    period: Optional[str] = None,
    prepost: Optional[bool] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Download OHLCV for a ticker at the requested candle size.
    Returns (dataframe with lowercase ohlcv, meta dict).
    """
    t = str(ticker).strip()
    c = normalize_candle(candle)
    spec = dict(CANDLE_SPECS[c])
    use_period = period or spec["period"]
    if c in ("1mo", "3mo", "6mo") and use_period in ("1y", "2y", "6mo", "3mo", "1mo"):
        use_period = spec["period"]
    interval = spec["interval"]
    use_prepost = spec.get("prepost", False) if prepost is None else prepost
    meta = {
        "ticker": t,
        "candle": c,
        "label": spec["label"],
        "period": use_period,
        "interval": interval,
        "resample": spec.get("resample"),
        "ann": spec["ann"],
        "error": None,
    }
    if not t:
        meta["error"] = "missing ticker"
        return pd.DataFrame(), meta

    df = pd.DataFrame()
    try:
        df = yf.download(
            tickers=t,
            period=use_period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            prepost=use_prepost,
            threads=False,
        )
        if df is None or df.empty:
            df = yf.Ticker(t).history(
                period=use_period,
                interval=interval,
                auto_adjust=True,
                prepost=use_prepost,
            )
    except Exception as e:
        meta["error"] = str(e)
        return pd.DataFrame(), meta

    df = _flatten_ohlcv(df if df is not None else pd.DataFrame(), t)
    rule = spec.get("resample")
    if rule and not df.empty:
        try:
            df = _resample_ohlcv(df, rule)
        except Exception:
            alt = {"QS": "QE", "2QS": "6MS", "6MS": "2QE"}.get(str(rule), "QE")
            df = _resample_ohlcv(df, alt)
    meta["bars"] = int(len(df))
    if df.empty:
        meta["error"] = meta["error"] or "no data"
    return df, meta


def chart_time(idx, candle: str) -> Any:
    """lightweight-charts time: YYYY-MM-DD for daily+, unix seconds for 5m."""
    c = normalize_candle(candle)
    if c == "5m":
        ts = pd.Timestamp(idx)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.timestamp())
    ts = pd.Timestamp(idx)
    return ts.strftime("%Y-%m-%d")


def to_candles(df: pd.DataFrame, candle: str = "24h") -> list:
    rows = []
    if df is None or df.empty:
        return rows
    c = normalize_candle(candle)
    for idx, row in df.iterrows():
        close = float(row["close"])
        rows.append({
            "time": chart_time(idx, c),
            "open": float(row["open"]) if "open" in row and pd.notna(row["open"]) else close,
            "high": float(row["high"]) if "high" in row and pd.notna(row["high"]) else close,
            "low": float(row["low"]) if "low" in row and pd.notna(row["low"]) else close,
            "close": close,
            "volume": float(row["volume"]) if "volume" in row and pd.notna(row["volume"]) else 0.0,
        })
    return rows
