"""
Multi-Stock Technical Analysis Scanner
======================================
Scans US and Indian equities using classical TA plus an institutional quant overlay
(momentum, volatility, Hurst, CAPM, VWAP, Ichimoku, …) to produce BUY or SELL only.
Long-term candles: 24h (daily), 1 month, 3 month, 6 month.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor

# Yahoo Finance news base (yfinance often returns protocol-relative or site-relative links)
_YAHOO_NEWS_ORIGIN = "https://finance.yahoo.com"


def normalize_news_url(raw: str) -> str:
    """
    Turn yfinance news URLs into absolute https URLs on the real publisher/Yahoo domain.
    Fixes protocol-relative //..., site-relative /news/..., and path-only links so clicks
    never resolve to our own site's origin.
    """
    u = (raw or "").strip()
    if not u:
        return ""
    if u.startswith("//"):
        return "https:" + u
    low = u.lower()
    if low.startswith(("http://", "https://")):
        scheme = low.split(":", 1)[0]
        if scheme in ("javascript", "data", "vbscript"):
            return ""
        return u
    if u.startswith("/"):
        return _YAHOO_NEWS_ORIGIN + u
    return urljoin(_YAHOO_NEWS_ORIGIN + "/", u)


def _extract_news_fields(n) -> Dict[str, str]:
    """
    Normalize a yfinance news item to {title, url, publisher, summary}.
    Supports BOTH the legacy flat schema ({title, link, publisher}) and the
    current nested schema ({content: {title, canonicalUrl: {url}, provider: {displayName}}}).
    Without this, modern yfinance returns empty titles/URLs and news sentiment is always 0.
    """
    if not isinstance(n, dict):
        return {"title": "", "url": "", "publisher": "", "summary": ""}
    content = n.get("content") if isinstance(n.get("content"), dict) else {}
    title = n.get("title") or content.get("title") or ""
    url = n.get("link") or n.get("url") or ""
    if not url:
        for key in ("canonicalUrl", "clickThroughUrl"):
            u = content.get(key)
            if isinstance(u, dict) and u.get("url"):
                url = u["url"]
                break
            if isinstance(u, str) and u:
                url = u
                break
    publisher = n.get("publisher") or n.get("source") or ""
    if not publisher:
        prov = content.get("provider")
        if isinstance(prov, dict):
            publisher = prov.get("displayName") or ""
    summary = (
        n.get("summary") or n.get("description")
        or content.get("summary") or content.get("description") or ""
    )
    return {
        "title": str(title), "url": str(url),
        "publisher": str(publisher), "summary": str(summary),
    }

# ============================================================
# INDIA STOCK UNIVERSES (NSE)
# ============================================================

# Nifty 50 - all 50 constituents (Wikipedia, as of 8 Dec 2025)
# Note: TMPV (Tata Motors Passenger Vehicles) replaced TATAMOTORS after the demerger.
NIFTY50 = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDIGO.NS", "INFY.NS", "ITC.NS",
    "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "MAXHEALTH.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TMPV.NS", "TATASTEEL.NS",
    "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS",
]
# Nifty Next 50 - official constituents (Wikipedia / NSE, as of 30 Sep 2025 rebalance).
# The previous list here wrongly repeated 19 Nifty 50 names and missed most real members.
NIFTY_NEXT50 = [
    "ABB.NS", "ADANIENSOL.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "AMBUJACEM.NS",
    "BAJAJHLDNG.NS", "BAJAJHFL.NS", "BANKBARODA.NS", "BPCL.NS", "BRITANNIA.NS",
    "BOSCHLTD.NS", "CANBK.NS", "CGPOWER.NS", "CHOLAFIN.NS", "DIVISLAB.NS",
    "DLF.NS", "DMART.NS", "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS",
    "HAL.NS", "HINDZINC.NS", "HYUNDAI.NS", "ICICIGI.NS", "INDHOTEL.NS",
    "IOC.NS", "NAUKRI.NS", "IRFC.NS", "JINDALSTEL.NS", "JSWENERGY.NS",
    "LICI.NS", "LODHA.NS", "LTIM.NS", "MAZDOCK.NS", "PIDILITIND.NS",
    "PFC.NS", "PNB.NS", "RECLTD.NS", "MOTHERSON.NS", "SHREECEM.NS",
    "SIEMENS.NS", "ENRIN.NS", "SOLARINDS.NS", "TATAPOWER.NS", "TORNTPHARM.NS",
    "TVSMOTOR.NS", "UNITDSPR.NS", "VBL.NS", "VEDL.NS", "ZYDUSLIFE.NS",
]

# Default scan: Nifty 50 + Nifty Next 50 (India only, de-duplicated)
DEFAULT_TICKERS = list(dict.fromkeys(NIFTY50 + NIFTY_NEXT50))


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (EWM with alpha=1/period) — matches standard charting platforms.
    A plain rolling mean overstates oscillation vs. the canonical definition."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple:
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


def compute_sma_crossover(close: pd.Series, short: int = 10, long: int = 50) -> tuple:
    sma_short = close.rolling(window=short).mean()
    sma_long = close.rolling(window=long).mean()
    return sma_short, sma_long


def compute_momentum(close: pd.Series, period: int = 10) -> pd.Series:
    return (close / close.shift(period) - 1) * 100


def compute_volume_sma_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    sma_vol = volume.rolling(window=period).mean()
    return volume / (sma_vol + 1e-10)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder smoothing (standard ATR), not a simple rolling mean
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _get_extended_indicators(high, low, close, volume):
    """Compute Stochastic, Williams %R, CCI, ADX, OBV, Stoch RSI, Pivots."""
    from indicators import (
        compute_stochastic,
        compute_williams_r,
        compute_cci,
        compute_adx,
        compute_obv,
        compute_stoch_rsi,
        compute_pivot_points,
    )
    out = {}
    try:
        sk, sd = compute_stochastic(high, low, close)
        out["stoch_k"] = float(sk.iloc[-1]) if len(sk) and not pd.isna(sk.iloc[-1]) else None
        out["stoch_d"] = float(sd.iloc[-1]) if len(sd) and not pd.isna(sd.iloc[-1]) else None
    except Exception:
        pass
    try:
        wr = compute_williams_r(high, low, close)
        out["williams_r"] = float(wr.iloc[-1]) if len(wr) and not pd.isna(wr.iloc[-1]) else None
    except Exception:
        pass
    try:
        cci = compute_cci(high, low, close)
        out["cci"] = float(cci.iloc[-1]) if len(cci) and not pd.isna(cci.iloc[-1]) else None
    except Exception:
        pass
    try:
        adx, pdi, mdi = compute_adx(high, low, close)
        out["adx"] = float(adx.iloc[-1]) if len(adx) and not pd.isna(adx.iloc[-1]) else None
    except Exception:
        pass
    try:
        obv = compute_obv(close, volume)
        if len(obv) >= 5:
            obv_slope = (obv.iloc[-1] - obv.iloc[-5]) / (abs(obv.iloc[-1]) + 1e-10)
            out["obv_trend"] = float(np.clip(obv_slope, -1, 1))
    except Exception:
        pass
    try:
        out["stoch_rsi"] = compute_stoch_rsi(close)
    except Exception:
        pass
    try:
        pivot, r1, r2, s1, s2 = compute_pivot_points(high, low, close)
        out["pivot"], out["r1"], out["s1"] = pivot, r1, s1
    except Exception:
        pass
    return out


# ============================================================
# COMPOSITE SIGNAL LOGIC (Multi-factor scoring)
# ============================================================

@dataclass
class SignalResult:
    ticker: str
    action: str
    score: float
    price: float
    change_pct: float
    rsi: Optional[float]
    macd_hist: Optional[float]
    bb_position: Optional[float]
    momentum_10d: Optional[float]
    volume_ratio: Optional[float]
    reasons: List[str]
    buy_zone: Optional[tuple]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    support: Optional[float]
    resistance: Optional[float]
    news: List[Dict]
    news_sentiment: Optional[float] = None
    news_impact: Optional[float] = None
    pattern_signals: Optional[List[str]] = None
    pattern_score: Optional[float] = None
    factors_used: Optional[List[str]] = None  # for RLHF feedback
    # Extended indicators
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    williams_r: Optional[float] = None
    cci: Optional[float] = None
    adx: Optional[float] = None
    obv_trend: Optional[float] = None
    stoch_rsi: Optional[float] = None
    pivot: Optional[float] = None
    r1: Optional[float] = None
    s1: Optional[float] = None
    # Target timeline
    target_achieve_days: Optional[int] = None
    target_achieve_date: Optional[str] = None
    error: Optional[str] = None
    # Which indicators were used for scoring (None = all). Same order as request after normalize.
    indicators_for_score: Optional[List[str]] = None
    market: Optional[str] = None
    currency: Optional[str] = None
    candle: Optional[str] = None
    quant: Optional[Dict] = None
    ml: Optional[Dict] = None
    forecast: Optional[Dict] = None


# Indicators that participate in the composite score (manual selection applies to these).
ALL_SCORING_INDICATORS = frozenset({
    "rsi", "macd", "bollinger", "sma", "momentum", "volume",
    "stochastic", "williams_r", "cci", "adx", "obv", "stoch_rsi",
    "quant", "ml",
})

# Maximum absolute contribution of each factor to the raw score (before RLHF weight).
# Used to normalize the composite into [-1, 1] as "consensus of fired evidence".
_FACTOR_CAPS = {
    "rsi": 0.8, "macd": 0.5, "bollinger": 0.6, "sma": 0.4, "momentum": 0.3,
    "volume": 0.2, "stochastic": 0.4, "williams_r": 0.35, "cci": 0.3,
    "adx": 0.25, "obv": 0.2, "stoch_rsi": 0.35, "pattern": 0.5, "news": 0.6,
    "quant": 0.55, "ml": 0.7,
}

INDICATOR_ALIASES = {
    "williams": "williams_r",
    "williams_percent_r": "williams_r",
    "stoch": "stochastic",
    "bb": "bollinger",
    "vol": "volume",
    "mom": "momentum",
}

# API / UI catalog (id, human label)
INDICATOR_CATALOG: List[Dict[str, str]] = [
    {"id": "rsi", "label": "RSI"},
    {"id": "macd", "label": "MACD"},
    {"id": "bollinger", "label": "Bollinger Bands"},
    {"id": "sma", "label": "SMA crossover (10/50)"},
    {"id": "momentum", "label": "10d momentum"},
    {"id": "volume", "label": "Volume vs avg"},
    {"id": "stochastic", "label": "Stochastic"},
    {"id": "williams_r", "label": "Williams %R"},
    {"id": "cci", "label": "CCI"},
    {"id": "adx", "label": "ADX + DI"},
    {"id": "obv", "label": "OBV trend"},
    {"id": "stoch_rsi", "label": "Stochastic RSI"},
    {"id": "pattern", "label": "Chart patterns"},
    {"id": "news", "label": "News sentiment"},
    {"id": "quant", "label": "Quant desk (momentum, Hurst, CAPM, vol, Ichimoku)"},
    {"id": "ml", "label": "Trained 5-day model (walk-forward evaluated)"},
]


def normalize_indicator_set(indicators: Optional[List[str]]) -> Optional[frozenset]:
    """
    None / empty / all-invalid → None (use full model).
    Otherwise return frozenset of canonical ids.
    """
    if not indicators:
        return None
    out = set()
    for x in indicators:
        k = str(x).strip().lower().replace(" ", "_").replace("-", "_")
        k = INDICATOR_ALIASES.get(k, k)
        if k in ALL_SCORING_INDICATORS:
            out.add(k)
    return frozenset(out) if out else None


def scoring_active(key: str, enabled: Optional[frozenset]) -> bool:
    return enabled is None or key in enabled


def _indicators_applied_list(enabled: Optional[frozenset]) -> List[str]:
    if enabled is None:
        return sorted(ALL_SCORING_INDICATORS)
    return sorted(enabled)


def _get_rlhf_weights() -> dict:
    """Load RLHF-adapted factor weights. Algorithm improves from feedback."""
    try:
        from rlhf import get_weights
        return get_weights()
    except Exception:
        return {}


def _get_bb_position(close: float, upper: float, mid: float, lower: float) -> float:
    """Returns -1 (at/below lower) to 1 (at/above upper). 0 = at middle."""
    if pd.isna(upper) or pd.isna(lower) or upper <= lower:
        return 0.0
    if close <= lower:
        return -1.0
    if close >= upper:
        return 1.0
    # Linear interpolate
    return 2.0 * (close - mid) / (upper - lower)


def _binary_action(score: float) -> str:
    """Always BUY or SELL: strong thresholds, else weak signals use score sign."""
    if score >= 0.35:
        return "BUY"
    if score <= -0.35:
        return "SELL"
    return "BUY" if score >= 0 else "SELL"


def analyze_ticker(
    ticker: str,
    period: str = "2y",
    indicators: Optional[List[str]] = None,
    candle: str = "24h",
    benchmark: Optional[pd.Series] = None,
) -> SignalResult:
    """
    Compute technical indicators + quant overlay and produce a composite BUY or SELL signal.
    ``indicators``: if set, only those factors contribute to the score (manual mode).
    ``candle``: 24h | 1mo | 3mo | 6mo
    """
    from ohlc import fetch_ohlcv, lookbacks, normalize_candle, CANDLE_SPECS
    from markets import currency_for, market_for

    candle = normalize_candle(candle)
    spec = CANDLE_SPECS[candle]
    min_bars = int(spec.get("min_bars", 20))
    applied_labels = _indicators_applied_list(normalize_indicator_set(indicators))
    reasons = []
    try:
        data, meta = fetch_ohlcv(ticker, candle=candle, period=period)
        if data.empty or len(data) < min_bars:
            return SignalResult(
                ticker=ticker, action="ERROR", score=0, price=0, change_pct=0,
                rsi=None, macd_hist=None, bb_position=None, momentum_10d=None,
                volume_ratio=None, reasons=[
                    f"Insufficient price history (<{min_bars} {candle} bars) — no signal"
                ],
                buy_zone=None, stop_loss=None, take_profit=None, support=None, resistance=None,
                news=[], news_sentiment=None, news_impact=None, pattern_signals=None, pattern_score=None,
                factors_used=None, error=meta.get("error") or "Not enough history",
                indicators_for_score=applied_labels,
                market=market_for(ticker), currency=currency_for(ticker), candle=candle,
            )

        close = data["close"]
        volume = data["volume"] if "volume" in data.columns else pd.Series(1.0, index=data.index)
        lb = lookbacks(candle, len(close))

        rsi_s = compute_rsi(close, lb["rsi"])
        rsi = rsi_s.iloc[-1] if len(rsi_s) else np.nan
        macd_line, sig_line, hist = compute_macd(close)
        macd_hist = hist.iloc[-1] if len(hist) > 0 and not pd.isna(hist.iloc[-1]) else np.nan

        bb_u, bb_m, bb_l = compute_bollinger(close, lb["bb"])
        bb_pos = _get_bb_position(close.iloc[-1], bb_u.iloc[-1], bb_m.iloc[-1], bb_l.iloc[-1])

        sma_10, sma_50 = compute_sma_crossover(close, lb["sma_s"], lb["sma_l"])
        mom = compute_momentum(close, lb["mom"]).iloc[-1] if len(close) >= lb["mom"] + 1 else np.nan
        vol_ratio = (
            compute_volume_sma_ratio(volume, lb["vol"]).iloc[-1]
            if len(volume) >= lb["vol"] + 1 else np.nan
        )

        price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else price
        change_pct = (price / prev_close - 1) * 100 if prev_close else 0

        high_series = data["high"] if "high" in data.columns else close
        low_series = data["low"] if "low" in data.columns else close

        # Extended indicators: Stochastic, Williams, CCI, ADX, OBV, Stoch RSI, Pivots
        ext = _get_extended_indicators(high_series, low_series, close, volume)
        stoch_k = ext.get("stoch_k")
        stoch_d = ext.get("stoch_d")
        williams_r = ext.get("williams_r")
        cci_val = ext.get("cci")
        adx_val = ext.get("adx")
        obv_trend = ext.get("obv_trend")
        stoch_rsi_val = ext.get("stoch_rsi")
        pivot_val = ext.get("pivot")
        r1_val = ext.get("r1")
        s1_val = ext.get("s1")

        # Fetch news early (needed for sentiment analysis)
        raw_news = []
        try:
            t = yf.Ticker(ticker)
            raw_news = getattr(t, "news", None) or (t.get_news() if callable(getattr(t, "get_news", None)) else []) or []
        except Exception:
            pass

        enabled = normalize_indicator_set(indicators)
        # ========== SCORING (RLHF-adaptive weights) ==========
        weights = _get_rlhf_weights()
        score = 0.0
        n_factors = 0
        factors_used = []

        # RSI: <30 bullish (+), >70 bearish (-)
        if scoring_active("rsi", enabled) and not pd.isna(rsi):
            w = weights.get("rsi", 1.0)
            if rsi < 30:
                score += 0.8 * w
                factors_used.append("rsi")
                reasons.append(f"RSI oversold ({rsi:.0f})")
            elif rsi < 40:
                score += 0.3 * w
                factors_used.append("rsi")
                reasons.append(f"RSI low ({rsi:.0f})")
            elif rsi > 70:
                score -= 0.8 * w
                factors_used.append("rsi")
                reasons.append(f"RSI overbought ({rsi:.0f})")
            elif rsi > 60:
                score -= 0.3 * w
                factors_used.append("rsi")
                reasons.append(f"RSI high ({rsi:.0f})")

        # MACD histogram: positive = bullish, negative = bearish
        if scoring_active("macd", enabled) and not pd.isna(macd_hist) and macd_hist != 0:
            w = weights.get("macd", 1.0)
            factors_used.append("macd")
            macd_norm = np.clip(macd_hist / (price * 0.01), -1, 1)
            score += macd_norm * 0.5 * w
            if macd_hist > 0:
                reasons.append("MACD bullish")
            else:
                reasons.append("MACD bearish")

        # Bollinger position: below lower = buy, above upper = sell
        if scoring_active("bollinger", enabled) and not np.isnan(bb_pos):
            w = weights.get("bollinger", 1.0)
            factors_used.append("bollinger")
            score -= bb_pos * 0.6 * w
            if bb_pos < -0.5:
                reasons.append("Price near lower Bollinger")
            elif bb_pos > 0.5:
                reasons.append("Price near upper Bollinger")

        # SMA crossover: short > long = bullish
        if scoring_active("sma", enabled) and len(close) >= lb["sma_l"] + 1:
            w = weights.get("sma", 1.0)
            factors_used.append("sma")
            s10, s50 = sma_10.iloc[-1], sma_50.iloc[-1]
            if s10 > s50:
                score += 0.4 * w
                reasons.append(f"SMA {lb['sma_s']} > SMA {lb['sma_l']}")
            else:
                score -= 0.4 * w
                reasons.append(f"SMA {lb['sma_s']} < SMA {lb['sma_l']}")

        # Momentum (10d): positive = bullish
        if scoring_active("momentum", enabled) and not pd.isna(mom):
            w = weights.get("momentum", 1.0)
            factors_used.append("momentum")
            mom_norm = np.clip(mom / 15, -1, 1)
            score += mom_norm * 0.3 * w
            if abs(mom) > 5:
                reasons.append(f"{lb['mom']}-bar momentum {mom:+.1f}%")

        # Volume confirmation: high vol on up move = stronger signal
        if scoring_active("volume", enabled) and not pd.isna(vol_ratio) and vol_ratio > 1.2:
            w = weights.get("volume", 1.0)
            factors_used.append("volume")
            if change_pct > 0:
                score += 0.2 * w
                reasons.append("Volume confirmation")
            else:
                score -= 0.2 * w
                reasons.append("High volume on decline")

        # Stochastic: <20 oversold (bullish), >80 overbought (bearish)
        if scoring_active("stochastic", enabled) and stoch_k is not None:
            w = weights.get("stochastic", 1.0)
            factors_used.append("stochastic")
            if stoch_k < 20:
                score += 0.4 * w
                reasons.append(f"Stochastic oversold ({stoch_k:.0f})")
            elif stoch_k > 80:
                score -= 0.4 * w
                reasons.append(f"Stochastic overbought ({stoch_k:.0f})")

        # Williams %R: < -80 oversold, > -20 overbought
        if scoring_active("williams_r", enabled) and williams_r is not None:
            w = weights.get("williams_r", 1.0)  # key must match factors_used / RLHF store
            factors_used.append("williams_r")
            if williams_r < -80:
                score += 0.35 * w
                reasons.append(f"Williams %R oversold ({williams_r:.0f})")
            elif williams_r > -20:
                score -= 0.35 * w
                reasons.append(f"Williams %R overbought ({williams_r:.0f})")

        # CCI: < -100 oversold, > 100 overbought
        if scoring_active("cci", enabled) and cci_val is not None:
            w = weights.get("cci", 1.0)
            factors_used.append("cci")
            if cci_val < -100:
                score += 0.3 * w
                reasons.append(f"CCI oversold ({cci_val:.0f})")
            elif cci_val > 100:
                score -= 0.3 * w
                reasons.append(f"CCI overbought ({cci_val:.0f})")

        # ADX: strong trend (>25) confirms direction; +DI > -DI bullish
        if scoring_active("adx", enabled) and adx_val is not None and adx_val > 20:
            try:
                from indicators import compute_adx
                _, pdi, mdi = compute_adx(high_series, low_series, close, 14)
                pdi_val = float(pdi.iloc[-1]) if len(pdi) else None
                mdi_val = float(mdi.iloc[-1]) if len(mdi) else None
                if pdi_val is not None and mdi_val is not None:
                    w = weights.get("adx", 1.0)
                    factors_used.append("adx")
                    if pdi_val > mdi_val:
                        score += 0.25 * w
                        reasons.append(f"ADX trend bullish ({adx_val:.0f})")
                    else:
                        score -= 0.25 * w
                        reasons.append(f"ADX trend bearish ({adx_val:.0f})")
            except Exception:
                pass

        # OBV trend: positive slope = bullish
        if scoring_active("obv", enabled) and obv_trend is not None and obv_trend != 0:
            w = weights.get("obv", 1.0)
            factors_used.append("obv")
            score += obv_trend * 0.2 * w
            if obv_trend > 0.3:
                reasons.append("OBV rising")
            elif obv_trend < -0.3:
                reasons.append("OBV falling")

        # Stoch RSI: <20 oversold, >80 overbought
        if scoring_active("stoch_rsi", enabled) and stoch_rsi_val is not None:
            w = weights.get("stoch_rsi", 1.0)
            factors_used.append("stoch_rsi")
            if stoch_rsi_val < 20:
                score += 0.35 * w
                reasons.append(f"Stoch RSI oversold ({stoch_rsi_val:.0f})")
            elif stoch_rsi_val > 80:
                score -= 0.35 * w
                reasons.append(f"Stoch RSI overbought ({stoch_rsi_val:.0f})")

        # Pattern analysis (always detect for display; score only if enabled)
        pattern_signals_list = []
        pattern_score_val = 0.0
        news_sentiment_val = 0.0
        news_impact_val = 0.0
        try:
            from patterns import detect_all_patterns
            ph = data["high"] if "high" in data.columns else close
            pl = data["low"] if "low" in data.columns else close
            support_pre = float(close.rolling(20).min().iloc[-1]) if len(close) >= 20 else price * 0.97
            resistance_pre = float(close.rolling(20).max().iloc[-1]) if len(close) >= 20 else price * 1.03
            pat_sigs, pattern_score_val = detect_all_patterns(
                data, close, support_pre, resistance_pre
            )
            pattern_signals_list = [f"{s.name}({s.direction})" for s in pat_sigs]
            if scoring_active("pattern", enabled) and pat_sigs:
                w = weights.get("pattern", 1.0)
                factors_used.append("pattern")
                score += pattern_score_val * 0.5 * w
                for s in pat_sigs:
                    if s.direction != 0:
                        reasons.append(f"Pattern: {s.name} (bearish)" if s.direction < 0 else f"Pattern: {s.name} (bullish)")
        except Exception:
            pass

        # News sentiment (always compute for display; score only if enabled)
        try:
            from news_analysis import analyze_news_sentiment
            news_list_for_analysis = []
            for n in (raw_news or []):
                f = _extract_news_fields(n)
                news_list_for_analysis.append({
                    "title": f["title"],
                    "description": f["summary"],
                    "content": "",
                })
            news_result = analyze_news_sentiment(news_list_for_analysis)
            news_sentiment_val = news_result["sentiment_score"]
            news_impact_val = news_result["impact"]
            if scoring_active("news", enabled) and news_impact_val > 0.2 and abs(news_sentiment_val) > 0.2:
                w = weights.get("news", 1.0)
                factors_used.append("news")
                score += news_sentiment_val * news_impact_val * 0.6 * w
                reasons.append(news_result["summary"])
        except Exception:
            pass

        quant_bundle = None
        try:
            from quant_engine import compute_quant_bundle
            quant_bundle = compute_quant_bundle(
                data, ticker, candle=candle, benchmark=benchmark
            )
            qscore = quant_bundle.get("quant_score")
            if scoring_active("quant", enabled) and qscore is not None:
                w = weights.get("quant", 1.0)
                factors_used.append("quant")
                score += float(qscore) * 0.55 * w
                for msg in (quant_bundle.get("quant_reasons") or [])[:4]:
                    reasons.append(msg)
        except Exception:
            quant_bundle = None

        ml_pred = None
        try:
            from pred_model import predict_last
            ml_pred = predict_last(data)
            if scoring_active("ml", enabled) and ml_pred and ml_pred.get("p_buy") is not None:
                w = weights.get("ml", 1.15)
                pb = float(ml_pred["p_buy"])
                # Only blend high-confidence calls; pooled OOS is barely above chance.
                if pb >= 0.62 or pb <= 0.38:
                    factors_used.append("ml")
                    edge = (pb - 0.5) * 2.0
                    score += edge * 0.7 * w
                oos = ml_pred.get("oos_accuracy")
                bit = f" · OOS hit {oos:.0%}" if isinstance(oos, (int, float)) and oos else ""
                reasons.append(f"Trained model P(lead {ml_pred.get('horizon', 5)}d)={pb:.2f}{bit}")
        except Exception:
            ml_pred = None

        # Normalize to [-1, 1]: raw score / max possible score of the factors that
        # actually fired = "evidence consensus". A shrinkage term damps signals
        # backed by only 1-2 factors so a single mild indicator can't look strong.
        n_fired = len(factors_used)
        max_contrib = sum(
            _FACTOR_CAPS.get(f, 0.5) * weights.get(f, 1.0) for f in factors_used
        )
        if n_fired > 0 and max_contrib > 1e-9:
            consensus = score / max_contrib
            shrink = n_fired / (n_fired + 2.0)
            score = float(np.clip(consensus * shrink, -1, 1))
        else:
            score = 0.0

        # BUY or SELL only (no HOLD — weak band maps by score sign)
        action = _binary_action(float(score))

        sr_n = max(6, lb["bb"])
        low_20 = float(close.rolling(sr_n).min().iloc[-1]) if len(close) >= sr_n else price * 0.97
        high_20 = float(close.rolling(sr_n).max().iloc[-1]) if len(close) >= sr_n else price * 1.03
        support = low_20
        resistance = high_20

        # ATR for stop placement
        atr = compute_atr(high_series, low_series, close, lb["atr"]).iloc[-1] if len(close) >= lb["atr"] + 1 else price * 0.02
        atr = float(atr) if not (pd.isna(atr) or atr <= 0) else price * 0.02

        # Buy zone, SL, TP by action
        if action == "BUY":
            buy_zone = (support, min(price * 1.01, support * 1.03))
            stop_loss = max(support - atr * 1.5, 0.01)
            take_profit = resistance + atr * 0.5
        else:
            buy_zone = None
            stop_loss = high_20 + atr * 1.5
            take_profit = max(support - atr * 0.5, 0.01)

        # Target timeline: estimate days to reach TP (or SL for SELL).
        # Use the 20-day average absolute daily move, not just yesterday's change
        # (a single day is far too noisy to extrapolate from).
        target_price = take_profit if action == "BUY" else stop_loss
        atr_pct = (atr / price) * 100
        recent_abs_ret = close.pct_change().abs().tail(20)
        daily_return_pct = float(recent_abs_ret.mean() * 100) if len(recent_abs_ret) else atr_pct
        if not np.isfinite(daily_return_pct) or daily_return_pct <= 0:
            daily_return_pct = atr_pct
        try:
            from indicators import compute_target_days
            target_days = compute_target_days(price, target_price, daily_return_pct, atr_pct)
        except Exception:
            target_days = max(5, min(30, int(20 * atr_pct)))
        from datetime import datetime, timedelta
        target_date = (datetime.now() + timedelta(days=target_days)).strftime("%Y-%m-%d")

        hist_pred = None
        try:
            from historical_predict import compute_historical_prediction
            fwd = max(1, int(spec.get("fwd_bars") or 5))
            hist_pred = compute_historical_prediction(
                data, close, high_series, low_series, support, resistance, fwd_days=fwd
            )
        except Exception:
            hist_pred = None
        hist_compact = None
        if isinstance(hist_pred, dict) and hist_pred.get("expected_fwd_return_pct") is not None:
            hist_compact = {
                "expected_fwd_return_pct": hist_pred.get("expected_fwd_return_pct"),
                "direction": hist_pred.get("direction"),
                "confidence": hist_pred.get("confidence"),
                "summary": (hist_pred.get("summary") or "")[:180],
            }
        from forecast import build_forecast
        forecast = build_forecast(
            price=price,
            action=action,
            score=score,
            stop_loss=stop_loss,
            take_profit=take_profit,
            support=support,
            resistance=resistance,
            atr=atr,
            target_days=target_days,
            ml=ml_pred if isinstance(ml_pred, dict) else None,
            hist=hist_compact,
            candle=candle,
            why=(reasons[0] if reasons else None),
        )

        # Build news list from raw_news — absolute external URLs only (see normalize_news_url)
        news_list = []
        for n in (raw_news or [])[:8]:
            f = _extract_news_fields(n)
            url = normalize_news_url(f["url"])
            title = f["title"].strip() or "Yahoo Finance article"
            news_list.append({
                "title": title[:120],
                "url": url,
                "publisher": f["publisher"],
            })

        return SignalResult(
            ticker=ticker,
            action=action,
            score=float(score),
            price=price,
            change_pct=float(change_pct),
            rsi=float(rsi) if not pd.isna(rsi) else None,
            macd_hist=float(macd_hist) if not pd.isna(macd_hist) else None,
            bb_position=float(bb_pos),
            momentum_10d=float(mom) if not pd.isna(mom) else None,
            volume_ratio=float(vol_ratio) if not pd.isna(vol_ratio) else None,
            reasons=reasons if reasons else ["No strong signals"],
            buy_zone=buy_zone,
            stop_loss=stop_loss,
            take_profit=take_profit,
            support=support,
            resistance=resistance,
            news=news_list,
            news_sentiment=news_sentiment_val,
            news_impact=news_impact_val,
            pattern_signals=pattern_signals_list if pattern_signals_list else None,
            pattern_score=round(pattern_score_val, 3) if pattern_score_val != 0 else None,
            factors_used=factors_used if factors_used else None,
            stochastic_k=stoch_k,
            stochastic_d=stoch_d,
            williams_r=williams_r,
            cci=cci_val,
            adx=adx_val,
            obv_trend=obv_trend,
            stoch_rsi=stoch_rsi_val,
            pivot=pivot_val,
            r1=r1_val,
            s1=s1_val,
            target_achieve_days=target_days,
            target_achieve_date=target_date,
            indicators_for_score=_indicators_applied_list(enabled),
            market=market_for(ticker),
            currency=currency_for(ticker),
            candle=candle,
            quant=quant_bundle,
            ml=ml_pred,
            forecast=forecast,
        )

    except Exception as e:
        from markets import currency_for as _cf, market_for as _mf
        return SignalResult(
            ticker=ticker, action="ERROR", score=0, price=0, change_pct=0,
            rsi=None, macd_hist=None, bb_position=None, momentum_10d=None,
            volume_ratio=None, reasons=[f"Error: {str(e)} — no signal"],
            buy_zone=None, stop_loss=None, take_profit=None, support=None, resistance=None,
            news=[], news_sentiment=None, news_impact=None, pattern_signals=None, pattern_score=None,
            factors_used=None, error=str(e),
            indicators_for_score=applied_labels,
            market=_mf(ticker), currency=_cf(ticker), candle=candle,
        )


def scan_tickers(
    tickers: Optional[List[str]] = None,
    period: str = "2y",
    filter_action: Optional[str] = None,
    indicators: Optional[List[str]] = None,
    candle: str = "24h",
    universe: Optional[str] = None,
) -> List[Dict]:
    """
    Scan a list of tickers and return signal results.
    filter_action: "BUY" | "SELL" | None (return all)
    indicators: optional subset of factor ids for scoring (manual mode); None = all.
    candle: 24h | 1mo | 3mo | 6mo
    """
    from ohlc import fetch_ohlcv, normalize_candle
    from markets import benchmark_symbol, universe_symbols
    from quant_engine import cross_section_ranks

    candle = normalize_candle(candle)
    if not tickers:
        tickers = universe_symbols(universe)
    tickers = [str(t).strip() for t in tickers if str(t).strip()]

    bench_cache: Dict[str, Optional[pd.Series]] = {}
    for sym in {benchmark_symbol(t) for t in tickers}:
        df, _ = fetch_ohlcv(sym, candle=candle, period=period)
        bench_cache[sym] = df["close"] if df is not None and not df.empty else None

    def _bench_for(t: str) -> Optional[pd.Series]:
        return bench_cache.get(benchmark_symbol(t))

    max_workers = min(8, max(1, len(tickers)))
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            analyzed = list(ex.map(
                lambda t: analyze_ticker(
                    t, period=period, indicators=indicators, candle=candle, benchmark=_bench_for(t)
                ),
                tickers,
            ))
    else:
        analyzed = [
            analyze_ticker(t, period=period, indicators=indicators, candle=candle, benchmark=_bench_for(t))
            for t in tickers
        ]

    compact_q = (
        "quant_score", "sharpe", "sortino", "calmar", "max_drawdown_pct", "var95_pct",
        "cvar95_pct", "beta", "alpha_ann_pct", "hurst", "hurst_read", "ou_halflife_bars",
        "momentum_skip_pct", "reversal_pct", "rel_strength_pct", "vol_yang_zhang",
        "vol_garch11", "vol_ewma", "vol_regime", "kelly_half", "efficiency_ratio",
        "amihud", "ichimoku", "supertrend", "quant_reasons",
    )

    results = []
    for r in analyzed:
        buy_zone_ser = [round(r.buy_zone[0], 2), round(r.buy_zone[1], 2)] if r.buy_zone else None
        q = None
        if r.quant and isinstance(r.quant, dict):
            q = {k: r.quant.get(k) for k in compact_q if k in r.quant}
        d = {
            "ticker": r.ticker,
            "action": r.action,
            "factors_used": r.factors_used,
            "score": round(r.score, 3),
            "price": r.price,
            "change_pct": round(r.change_pct, 2),
            "rsi": round(r.rsi, 1) if r.rsi is not None else None,
            "macd_hist": round(r.macd_hist, 4) if r.macd_hist is not None else None,
            "bb_position": round(r.bb_position, 2) if r.bb_position is not None else None,
            "momentum_10d": round(r.momentum_10d, 2) if r.momentum_10d is not None else None,
            "volume_ratio": round(r.volume_ratio, 2) if r.volume_ratio is not None else None,
            "reasons": r.reasons,
            "buy_zone": buy_zone_ser,
            "stop_loss": round(r.stop_loss, 2) if r.stop_loss is not None else None,
            "take_profit": round(r.take_profit, 2) if r.take_profit is not None else None,
            "support": round(r.support, 2) if r.support is not None else None,
            "resistance": round(r.resistance, 2) if r.resistance is not None else None,
            "news": r.news,
            "news_sentiment": round(r.news_sentiment, 3) if r.news_sentiment is not None else None,
            "news_impact": round(r.news_impact, 2) if r.news_impact is not None else None,
            "pattern_signals": r.pattern_signals,
            "pattern_score": r.pattern_score,
            "stochastic_k": round(r.stochastic_k, 1) if r.stochastic_k is not None else None,
            "stochastic_d": round(r.stochastic_d, 1) if r.stochastic_d is not None else None,
            "williams_r": round(r.williams_r, 1) if r.williams_r is not None else None,
            "cci": round(r.cci, 1) if r.cci is not None else None,
            "adx": round(r.adx, 1) if r.adx is not None else None,
            "obv_trend": round(r.obv_trend, 2) if r.obv_trend is not None else None,
            "stoch_rsi": round(r.stoch_rsi, 1) if r.stoch_rsi is not None else None,
            "pivot": round(r.pivot, 2) if r.pivot is not None else None,
            "r1": round(r.r1, 2) if r.r1 is not None else None,
            "s1": round(r.s1, 2) if r.s1 is not None else None,
            "target_achieve_days": r.target_achieve_days,
            "target_achieve_date": r.target_achieve_date,
            "error": r.error,
            "indicators_for_score": r.indicators_for_score,
            "market": r.market,
            "currency": r.currency,
            "candle": r.candle,
            "quant": q,
            "ml": r.ml,
            "forecast": r.forecast,
        }
        if filter_action and r.action != filter_action:
            continue
        results.append(d)
    results = cross_section_ranks(results)
    return results
