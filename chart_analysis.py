"""
Live chart payload: OHLCV + pattern overlays for the frontend chart.
Supports long-term candles 24h / 1mo / 3mo / 6mo.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from patterns import detect_all_patterns
from historical_predict import compute_historical_prediction
from ohlc import fetch_ohlcv, lookbacks, normalize_candle, to_candles
from markets import currency_for, market_for


def _swing_points(
    high: pd.Series, low: pd.Series, order: int = 4
) -> Tuple[List[int], List[int]]:
    """Local maxima on high, local minima on low."""
    hi = high.values
    lo = low.values
    n = len(high)
    swing_hi: List[int] = []
    swing_lo: List[int] = []
    order = max(2, min(order, max(2, n // 8)))
    for i in range(order, n - order):
        if hi[i] >= np.nanmax(hi[i - order : i + order + 1]):
            swing_hi.append(i)
        if lo[i] <= np.nanmin(lo[i - order : i + order + 1]):
            swing_lo.append(i)
    return swing_hi, swing_lo


def build_chart_payload(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    fwd_days: int = 5,
    candle: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch OHLCV and return JSON for Lightweight Charts + pattern analysis.
    Prefer ``candle`` (24h, 1mo, 3mo, 6mo); interval/period are fallbacks.
    """
    t = str(ticker).strip()
    if not t:
        return {"error": "missing ticker", "ticker": ticker}

    cndl = normalize_candle(candle or "24h")
    use_period = period
    if cndl in ("1mo", "3mo", "6mo") and (not period or period in ("1y", "2y", "6mo", "3mo")):
        use_period = "max"
    data, meta = fetch_ohlcv(t, candle=cndl, period=use_period)
    if data is None or data.empty or len(data) < 8:
        return {"error": meta.get("error") or "not enough data", "ticker": t}

    close = data["close"]
    high = data["high"] if "high" in data.columns else close
    low = data["low"] if "low" in data.columns else close
    lb = lookbacks(cndl, len(close))
    sr_n = max(4, lb["bb"])

    support = float(close.rolling(sr_n).min().iloc[-1]) if len(close) >= sr_n else float(close.iloc[-1]) * 0.97
    resistance = float(close.rolling(sr_n).max().iloc[-1]) if len(close) >= sr_n else float(close.iloc[-1]) * 1.03

    merged, composite = detect_all_patterns(data, close, support, resistance)
    composite = float(max(-1.0, min(1.0, composite)))

    candles = to_candles(data, cndl)
    idx_list = list(data.index)
    swing_hi, swing_lo = _swing_points(high, low, order=max(2, min(4, len(close) // 15)))
    markers: List[Dict[str, Any]] = []
    for j in swing_hi[-6:]:
        if 0 <= j < len(idx_list) and 0 <= j < len(candles):
            markers.append(
                {
                    "time": candles[j]["time"],
                    "position": "aboveBar",
                    "color": "#f87171",
                    "shape": "arrowDown",
                    "text": "SH",
                }
            )
    for j in swing_lo[-6:]:
        if 0 <= j < len(idx_list) and 0 <= j < len(candles):
            markers.append(
                {
                    "time": candles[j]["time"],
                    "position": "belowBar",
                    "color": "#4ade80",
                    "shape": "arrowUp",
                    "text": "SL",
                }
            )

    price_lines = [
        {"price": support, "color": "#22c55e", "title": f"Support ~{support:.2f}"},
        {"price": resistance, "color": "#ef4444", "title": f"Resistance ~{resistance:.2f}"},
    ]

    pattern_payload = [
        {
            "name": s.name,
            "direction": s.direction,
            "strength": round(s.strength, 3),
            "confidence": round(s.confidence, 3),
        }
        for s in merged
    ]

    fwd = int(fwd_days) if fwd_days else int(lb["fwd_bars"])
    try:
        historical_prediction = compute_historical_prediction(
            data, close, high, low, support, resistance, fwd_days=max(1, fwd)
        )
    except Exception as exc:
        historical_prediction = {"error": str(exc), "summary": "Historical model unavailable."}

    return {
        "ticker": t,
        "period": meta.get("period") or period,
        "interval": meta.get("interval") or interval,
        "candle": cndl,
        "market": market_for(t),
        "currency": currency_for(t),
        "candles": candles,
        "markers": markers,
        "price_lines": price_lines,
        "support": support,
        "resistance": resistance,
        "pattern_signals": pattern_payload,
        "composite_pattern_score": composite,
        "historical_prediction": historical_prediction,
        "last_close": float(close.iloc[-1]),
        "bar_count": len(candles),
    }
