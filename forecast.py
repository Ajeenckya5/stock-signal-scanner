"""Per-name expected move: direction, size, range, and P(up). Educational, not advice."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

HORIZON_BARS = {"24h": 5, "1d": 5, "1mo": 3, "3mo": 2, "6mo": 2, "5m": 12}
HORIZON_LABEL = {
    "24h": "5 sessions",
    "1d": "5 sessions",
    "1mo": "3 months",
    "3mo": "2 quarters",
    "6mo": "2 half-years",
    "5m": "this session",
}


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _round(x: Optional[float], n: int = 2) -> Optional[float]:
    v = _finite(x)
    return None if v is None else round(v, n)


def p_up_from_row(action: Optional[str], score: Any, ml: Optional[Dict[str, Any]]) -> float:
    ml = ml or {}
    pb = _finite(ml.get("p_buy"))
    if pb is not None:
        return max(0.05, min(0.95, pb))
    sc = _finite(score) or 0.0
    p = 0.5 + 0.35 * math.tanh(sc * 1.8)
    if action == "BUY":
        p = max(p, 0.52)
    elif action == "SELL":
        p = min(p, 0.48)
    return max(0.12, min(0.88, p))


def build_forecast(
    *,
    price: Any,
    action: Optional[str],
    score: Any = None,
    stop_loss: Any = None,
    take_profit: Any = None,
    support: Any = None,
    resistance: Any = None,
    atr: Any = None,
    target_days: Any = None,
    ml: Optional[Dict[str, Any]] = None,
    hist: Optional[Dict[str, Any]] = None,
    candle: str = "24h",
    why: Optional[str] = None,
) -> Dict[str, Any]:
    px = _finite(price)
    tp = _finite(take_profit)
    sl = _finite(stop_loss)
    sup = _finite(support)
    res = _finite(resistance)
    atr_v = _finite(atr)
    if atr_v is None and px and sl:
        atr_v = abs(px - sl) / 1.5
    atr_pct = (atr_v / px * 100.0) if px and atr_v else None

    p_up = p_up_from_row(action, score, ml)

    tech_pct = ((tp - px) / px * 100.0) if px and tp else None
    hist_pct = _finite((hist or {}).get("expected_fwd_return_pct"))
    ml_pct = ((2.0 * p_up - 1.0) * 1.15 * atr_pct) if atr_pct is not None else None

    weighted: list[tuple[float, float]] = []
    if tech_pct is not None:
        weighted.append((0.45, tech_pct))
    if hist_pct is not None:
        weighted.append((0.35, hist_pct))
    if ml_pct is not None:
        weighted.append((0.20, ml_pct))
    if weighted:
        expected = sum(w * v for w, v in weighted) / sum(w for w, _ in weighted)
    else:
        expected = 2.4 if (action == "BUY" or p_up >= 0.5) else -2.4

    # List membership follows the predicted price move, not the ML vote alone.
    if expected > 0.05:
        direction = "up"
    elif expected < -0.05:
        direction = "down"
    else:
        direction = "up" if (action == "BUY" or p_up >= 0.5) else "down"

    pad = atr_pct * 0.7 if atr_pct is not None else abs(expected) * 0.55
    res_pct = ((res - px) / px * 100.0) if px and res else None
    sup_pct = ((sup - px) / px * 100.0) if px and sup else None
    sl_pct = ((sl - px) / px * 100.0) if px and sl else None
    highs = [v for v in (expected, tech_pct, res_pct, pad) if v is not None]
    lows = [v for v in (expected, tech_pct if tech_pct is not None and tech_pct < 0 else None, sup_pct, sl_pct, -pad) if v is not None]
    high_pct = max(highs) if highs else pad
    low_pct = min(lows) if lows else -pad
    if high_pct < expected:
        high_pct = expected + pad * 0.4
    if low_pct > expected:
        low_pct = expected - pad * 0.4

    if direction == "up":
        target = tp if (tp and px and tp >= px) else (res if res else (px * (1 + abs(expected) / 100.0) if px else None))
    else:
        target = tp if (tp and px and tp <= px) else (sup if sup else (px * (1 - abs(expected) / 100.0) if px else None))

    ml_conf = _finite((ml or {}).get("confidence"))
    hist_conf = _finite((hist or {}).get("confidence"))
    score_conf = abs(_finite(score) or 0.0)
    confidence = max(c for c in (ml_conf or 0.0, hist_conf or 0.0, score_conf) )

    note = why or (hist or {}).get("summary") or ""
    if isinstance(note, str) and len(note) > 140:
        note = note[:137] + "…"

    return {
        "direction": direction,
        "p_up": _round(p_up, 3),
        "expected_pct": _round(expected, 2),
        "low_pct": _round(low_pct, 2),
        "high_pct": _round(high_pct, 2),
        "target": _round(target, 2),
        "stop": _round(sl, 2),
        "horizon": HORIZON_LABEL.get(candle or "24h", "5 sessions"),
        "horizon_bars": HORIZON_BARS.get(candle or "24h", 5),
        "horizon_days": int(target_days) if target_days is not None else None,
        "confidence": _round(confidence, 3),
        "why": note,
    }


def derive_forecast(row: Dict[str, Any]) -> Dict[str, Any]:
    existing = row.get("forecast")
    if isinstance(existing, dict) and existing.get("expected_pct") is not None:
        out = dict(existing)
        exp = _finite(out.get("expected_pct")) or 0.0
        out["direction"] = "up" if exp >= 0 else "down"
        return out
    reasons = row.get("reasons") or []
    why = reasons[0] if reasons else None
    return build_forecast(
        price=row.get("price"),
        action=row.get("action"),
        score=row.get("score"),
        stop_loss=row.get("stop_loss"),
        take_profit=row.get("take_profit"),
        support=row.get("support"),
        resistance=row.get("resistance"),
        atr=None,
        target_days=row.get("target_achieve_days"),
        ml=row.get("ml") if isinstance(row.get("ml"), dict) else None,
        hist=None,
        candle=row.get("candle") or "24h",
        why=why,
    )
