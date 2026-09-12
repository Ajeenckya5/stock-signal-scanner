"""
24/7 desk autopilot: market clocks, index pulse, scheduled scans, insight briefs.

Runs in a daemon thread for the life of the FastAPI process. The UI polls
/desk/snapshot — users can still trigger a manual scan or change what is watched.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, time as dtime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

_DIR = os.path.dirname(__file__)
_CFG_PATH = os.path.join(_DIR, ".desk_config.json")
_SNAP_PATH = os.path.join(_DIR, ".desk_snapshot.json")

MAX_AUTOPILOT = 160
PULSE_SYMBOLS = ["SPY", "QQQ", "IWM", "^NSEI", "^NSEBANK"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "universe": "liquid",
    "candle": "24h",
    "interval_sec": 900,
    "watchlist": [],
    "indicators": None,
    "intraday_enabled": True,
    "max_names": 80,
}

_lock = threading.RLock()
_wake = threading.Event()
_stop = threading.Event()
_thread: Optional[threading.Thread] = None
_config: Dict[str, Any] = dict(DEFAULT_CONFIG)
_state: Dict[str, Any] = {
    "running": False,
    "last_error": None,
    "last_scan_at": None,
    "last_pulse_at": None,
    "last_intraday_at": None,
    "next_scan_at": None,
    "scan_in_progress": False,
    "names": 0,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str, fallback):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return fallback


def _save_json(path: str, data) -> None:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        pass


def _session(name: str, tz_name: str, open_t: dtime, close_t: dtime, pre: Optional[dtime] = None, post: Optional[dtime] = None) -> Dict[str, Any]:
    now = datetime.now(ZoneInfo(tz_name))
    t = now.time()
    weekday = now.weekday() < 5
    status = "closed"
    if weekday:
        if open_t <= t <= close_t:
            status = "open"
        elif pre and pre <= t < open_t:
            status = "pre"
        elif post and close_t < t <= post:
            status = "after"
    return {
        "id": name,
        "name": "NYSE / Nasdaq" if name == "US" else "NSE",
        "tz": tz_name,
        "local_time": now.strftime("%H:%M:%S"),
        "local_date": now.strftime("%a %d %b"),
        "status": status,
        "open": open_t.strftime("%H:%M"),
        "close": close_t.strftime("%H:%M"),
        "weekday": weekday,
    }


def market_clocks() -> Dict[str, Any]:
    us = _session("US", "America/New_York", dtime(9, 30), dtime(16, 0), dtime(4, 0), dtime(20, 0))
    india = _session("IN", "Asia/Kolkata", dtime(9, 15), dtime(15, 30))
    utc = datetime.now(timezone.utc)
    any_open = us["status"] == "open" or india["status"] == "open"
    return {
        "utc": utc.strftime("%H:%M:%S"),
        "utc_iso": utc.isoformat(),
        "sessions": [us, india],
        "any_cash_open": any_open,
        "desk": "live",
    }


def _compact_row(r: Dict[str, Any]) -> Dict[str, Any]:
    q = r.get("quant") or {}
    news = r.get("news") or []
    return {
        "ticker": r.get("ticker"),
        "action": r.get("action"),
        "score": r.get("score"),
        "price": r.get("price"),
        "change_pct": r.get("change_pct"),
        "rsi": r.get("rsi"),
        "adx": r.get("adx"),
        "momentum_10d": r.get("momentum_10d"),
        "news_sentiment": r.get("news_sentiment"),
        "reasons": (r.get("reasons") or [])[:6],
        "stop_loss": r.get("stop_loss"),
        "take_profit": r.get("take_profit"),
        "support": r.get("support"),
        "resistance": r.get("resistance"),
        "pattern_signals": r.get("pattern_signals"),
        "market": r.get("market"),
        "currency": r.get("currency"),
        "candle": r.get("candle"),
        "factors_used": r.get("factors_used"),
        "target_achieve_days": r.get("target_achieve_days"),
        "target_achieve_date": r.get("target_achieve_date"),
        "error": r.get("error"),
        "news": [{"title": n.get("title"), "url": n.get("url")} for n in news[:3] if isinstance(n, dict)],
        "quant": {
            "quant_score": q.get("quant_score"),
            "sharpe": q.get("sharpe"),
            "hurst": q.get("hurst"),
            "hurst_read": q.get("hurst_read"),
            "beta": q.get("beta"),
            "momentum_skip_pct": q.get("momentum_skip_pct"),
            "rel_strength_pct": q.get("rel_strength_pct"),
            "vol_regime": q.get("vol_regime"),
            "kelly_half": q.get("kelly_half"),
            "cs_composite": q.get("cs_composite"),
            "max_drawdown_pct": q.get("max_drawdown_pct"),
        },
        "ml": r.get("ml"),
    }


def build_insights(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in results if (r.get("action") or "") in ("BUY", "SELL")]
    buys = sorted([r for r in ok if r["action"] == "BUY"], key=lambda x: float(x.get("score") or 0), reverse=True)
    sells = sorted([r for r in ok if r["action"] == "SELL"], key=lambda x: float(x.get("score") or 0))
    scores = [float(r.get("score") or 0) for r in ok]
    avg = float(sum(scores) / len(scores)) if scores else 0.0
    if avg > 0.08:
        tilt = "bullish"
    elif avg < -0.08:
        tilt = "bearish"
    else:
        tilt = "mixed"
    conv = [r for r in ok if abs(float(r.get("score") or 0)) >= 0.35]
    trending = mean_rev = high_vol = 0
    alerts: List[str] = []
    for r in ok:
        q = r.get("quant") or {}
        hr = q.get("hurst_read")
        if hr == "trending":
            trending += 1
        elif hr == "mean_reverting":
            mean_rev += 1
        if q.get("vol_regime") == "high_vol":
            high_vol += 1
        tkr = r.get("ticker")
        sc = float(r.get("score") or 0)
        if abs(sc) >= 0.4:
            alerts.append(f"{tkr} high-conviction {r.get('action')} ({sc:+.2f})")
        rsi = r.get("rsi")
        if rsi is not None and rsi >= 75:
            alerts.append(f"{tkr} RSI stretched ({rsi:.0f})")
        elif rsi is not None and rsi <= 25:
            alerts.append(f"{tkr} RSI washed-out ({rsi:.0f})")
        ns = r.get("news_sentiment")
        if ns is not None and abs(float(ns)) >= 0.45:
            alerts.append(f"{tkr} news impulse {float(ns):+.2f}")

    top_b = ", ".join(x["ticker"] for x in buys[:3]) or "—"
    top_s = ", ".join(x["ticker"] for x in sells[:3]) or "—"
    if not ok:
        headline = "Desk is warming up — first background pass has no names yet."
    elif tilt == "bullish":
        headline = f"Book tilts bullish. Highest-conviction longs: {top_b}."
    elif tilt == "bearish":
        headline = f"Book tilts defensive. Highest-conviction shorts: {top_s}."
    else:
        headline = f"Mixed tape — {len(buys)} longs vs {len(sells)} shorts. Watch {top_b} / {top_s}."

    def slim(rows, n=5):
        out = []
        for r in rows[:n]:
            out.append({
                "ticker": r.get("ticker"),
                "action": r.get("action"),
                "score": r.get("score"),
                "price": r.get("price"),
                "change_pct": r.get("change_pct"),
                "currency": r.get("currency"),
                "market": r.get("market"),
                "why": (r.get("reasons") or [""])[0],
            })
        return out

    return {
        "headline": headline,
        "tilt": tilt,
        "avg_score": round(avg, 3),
        "buy_count": len(buys),
        "sell_count": len(sells),
        "error_count": sum(1 for r in results if r.get("action") == "ERROR"),
        "high_conviction": len(conv),
        "trending": trending,
        "mean_reverting": mean_rev,
        "high_vol": high_vol,
        "top_buys": slim(buys),
        "top_sells": slim(sells),
        "conviction": slim(sorted(conv, key=lambda x: abs(float(x.get("score") or 0)), reverse=True), 8),
        "alerts": alerts[:12],
        "updated_at": _now_iso(),
    }


def _watch_symbols(cfg: Dict[str, Any]) -> List[str]:
    from markets import universe_symbols

    names = universe_symbols(cfg.get("universe") or "liquid")
    extra = [str(x).strip().upper() for x in (cfg.get("watchlist") or []) if str(x).strip()]
    merged = list(dict.fromkeys(extra + names))
    cap = int(cfg.get("max_names") or MAX_AUTOPILOT)
    cap = max(8, min(cap, MAX_AUTOPILOT))
    return merged[:cap]


def _run_long_scan(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    from ohlc import normalize_candle
    from scanner import scan_tickers

    candle = normalize_candle(cfg.get("candle") or "24h")
    period = "max" if candle in ("1mo", "3mo", "6mo") else "2y"
    tickers = _watch_symbols(cfg)
    inds = cfg.get("indicators") or None
    rows = scan_tickers(
        tickers=tickers,
        period=period,
        candle=candle,
        indicators=inds,
        universe=cfg.get("universe"),
    )
    return [_compact_row(r) for r in rows]


def _run_intraday(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    from intraday_engine import scan_intraday

    names = _watch_symbols(cfg)[:16]
    rows = scan_intraday(names)
    out = []
    for r in rows:
        out.append({
            "ticker": r.get("ticker"),
            "action": r.get("action"),
            "score": r.get("score"),
            "price": r.get("price"),
            "change_pct": r.get("change_pct"),
            "rsi": r.get("rsi"),
            "vwap": r.get("vwap"),
            "rvol": r.get("rvol"),
            "orh": r.get("orh"),
            "orl": r.get("orl"),
            "market": r.get("market"),
            "currency": r.get("currency"),
            "reasons": (r.get("reasons") or [])[:4],
            "error": r.get("error"),
        })
    return out


def _close_series(raw, sym):
    import pandas as pd

    try:
        if raw is None or getattr(raw, "empty", True):
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            if sym not in raw.columns.get_level_values(0):
                return None
            block = raw[sym]
            col = "Close" if "Close" in block.columns else "close"
            close = block[col]
        else:
            col = "Close" if "Close" in raw.columns else "close"
            close = raw[col]
        close = pd.to_numeric(close, errors="coerce").dropna()
        return close if len(close) >= 2 else None
    except Exception:
        return None


def _run_pulse() -> List[Dict[str, Any]]:
    import pandas as pd
    import yfinance as yf

    labels = {
        "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
        "^NSEI": "Nifty 50", "^NSEBANK": "Bank Nifty",
    }
    raw = None
    try:
        raw = yf.download(
            tickers=PULSE_SYMBOLS,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
            group_by="ticker",
        )
    except Exception:
        raw = None

    out = []
    missing = []
    for sym in PULSE_SYMBOLS:
        close = _close_series(raw, sym)
        if close is None:
            missing.append(sym)
            continue
        px = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        chg = (px / prev - 1.0) * 100 if prev else 0.0
        out.append({
            "symbol": sym,
            "label": labels.get(sym, sym),
            "price": round(px, 2),
            "change_pct": round(chg, 2),
        })
    for sym in missing:
        try:
            hist = yf.Ticker(sym).history(period="5d", interval="1d", auto_adjust=True)
            close = pd.to_numeric(hist["Close"] if "Close" in hist.columns else hist["close"], errors="coerce").dropna()
            if len(close) < 2:
                continue
            px = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            chg = (px / prev - 1.0) * 100 if prev else 0.0
            out.append({
                "symbol": sym,
                "label": labels.get(sym, sym),
                "price": round(px, 2),
                "change_pct": round(chg, 2),
            })
        except Exception:
            continue
    return out


def get_config() -> Dict[str, Any]:
    with _lock:
        return dict(_config)


def update_config(patch: Dict[str, Any]) -> Dict[str, Any]:
    global _config
    with _lock:
        cfg = dict(_config)
        if "enabled" in patch and patch["enabled"] is not None:
            cfg["enabled"] = bool(patch["enabled"])
        if patch.get("universe"):
            cfg["universe"] = str(patch["universe"])
        if patch.get("candle"):
            from ohlc import normalize_candle
            cfg["candle"] = normalize_candle(patch["candle"])
        if patch.get("interval_sec") is not None:
            cfg["interval_sec"] = int(max(120, min(int(patch["interval_sec"]), 7200)))
        if "watchlist" in patch and patch["watchlist"] is not None:
            cfg["watchlist"] = [str(x).strip().upper() for x in patch["watchlist"] if str(x).strip()][:40]
        if "indicators" in patch:
            cfg["indicators"] = patch["indicators"]
        if "intraday_enabled" in patch and patch["intraday_enabled"] is not None:
            cfg["intraday_enabled"] = bool(patch["intraday_enabled"])
        if patch.get("max_names") is not None:
            cfg["max_names"] = int(max(8, min(int(patch["max_names"]), MAX_AUTOPILOT)))
        _config = cfg
        _save_json(_CFG_PATH, cfg)
    _wake.set()
    return get_config()


def get_snapshot() -> Dict[str, Any]:
    snap = _load_json(_SNAP_PATH, {})
    with _lock:
        state = dict(_state)
        cfg = dict(_config)
    clocks = market_clocks()
    return {
        "config": cfg,
        "state": state,
        "clocks": clocks,
        "pulse": snap.get("pulse") or [],
        "results": snap.get("results") or [],
        "insights": snap.get("insights") or build_insights([]),
        "intraday": snap.get("intraday") or [],
        "intraday_insights": snap.get("intraday_insights") or None,
    }


def get_status() -> Dict[str, Any]:
    with _lock:
        state = dict(_state)
        cfg = dict(_config)
    return {"config": cfg, "state": state, "clocks": market_clocks()}


def _write_snapshot(**kwargs) -> None:
    snap = _load_json(_SNAP_PATH, {})
    snap.update(kwargs)
    snap["written_at"] = _now_iso()
    _save_json(_SNAP_PATH, snap)


def run_cycle(kind: str = "full") -> Dict[str, Any]:
    cfg = get_config()
    with _lock:
        _state["scan_in_progress"] = True
        _state["last_error"] = None
    try:
        if kind in ("pulse", "full"):
            pulse = _run_pulse()
            _write_snapshot(pulse=pulse)
            with _lock:
                _state["last_pulse_at"] = _now_iso()
        if kind in ("long", "full") and cfg.get("enabled"):
            rows = _run_long_scan(cfg)
            insights = build_insights(rows)
            _write_snapshot(results=rows, insights=insights)
            with _lock:
                _state["last_scan_at"] = _now_iso()
                _state["names"] = len(rows)
                _state["next_scan_at"] = datetime.fromtimestamp(
                    time.time() + int(cfg.get("interval_sec") or 900), tz=timezone.utc
                ).isoformat()
        if kind in ("intraday", "full") and cfg.get("enabled") and cfg.get("intraday_enabled"):
            clocks = market_clocks()
            if clocks.get("any_cash_open") or kind == "intraday":
                intra = _run_intraday(cfg)
                _write_snapshot(intraday=intra, intraday_insights=build_insights(intra))
                with _lock:
                    _state["last_intraday_at"] = _now_iso()
    except Exception as e:
        with _lock:
            _state["last_error"] = str(e)
    finally:
        with _lock:
            _state["scan_in_progress"] = False
    return get_snapshot()


def request_run(kind: str = "full") -> None:
    """Wake the loop and request an immediate cycle."""
    with _lock:
        _state["_force_kind"] = kind
    _wake.set()


def _loop() -> None:
    next_pulse = 0.0
    next_scan = 0.0
    next_intra = 0.0
    with _lock:
        _state["running"] = True
    # First pass: pulse quickly, then a full scan so the dashboard is not empty.
    try:
        run_cycle("pulse")
        cfg = get_config()
        if cfg.get("enabled"):
            run_cycle("full")
            next_scan = time.time() + int(cfg.get("interval_sec") or 900)
            next_intra = time.time() + 300
        next_pulse = time.time() + 120
    except Exception as e:
        with _lock:
            _state["last_error"] = str(e)

    while not _stop.is_set():
        cfg = get_config()
        now = time.time()
        force = None
        with _lock:
            force = _state.pop("_force_kind", None)
        try:
            if force:
                run_cycle("full" if force == "full" else force)
                if force in ("full", "long"):
                    next_scan = now + int(cfg.get("interval_sec") or 900)
            else:
                if now >= next_pulse:
                    run_cycle("pulse")
                    next_pulse = now + 120
                if cfg.get("enabled") and now >= next_scan:
                    run_cycle("long")
                    next_scan = now + int(cfg.get("interval_sec") or 900)
                clocks = market_clocks()
                if cfg.get("enabled") and cfg.get("intraday_enabled") and clocks.get("any_cash_open") and now >= next_intra:
                    run_cycle("intraday")
                    next_intra = now + 300
        except Exception as e:
            with _lock:
                _state["last_error"] = str(e)
        _wake.clear()
        _wake.wait(timeout=2.0)
    with _lock:
        _state["running"] = False


def start() -> None:
    global _thread, _config
    with _lock:
        loaded = _load_json(_CFG_PATH, None)
        if isinstance(loaded, dict):
            cfg = dict(DEFAULT_CONFIG)
            cfg.update({k: loaded[k] for k in DEFAULT_CONFIG if k in loaded})
            _config = cfg
        if _thread and _thread.is_alive():
            return
        _stop.clear()
        _thread = threading.Thread(target=_loop, name="predi-autopilot", daemon=True)
        _thread.start()


def stop() -> None:
    _stop.set()
    _wake.set()
