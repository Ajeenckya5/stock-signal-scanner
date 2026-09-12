"""
Walk-forward trained 5-day direction model for the research desk.

Uses the largest liquid US + India set Yahoo will give with daily history
from 2004 (Nifty 50 + S&P 100, cap 160). Labels are whether the name beats
that session's median next-5-session return (cross-section, not market drift).
Evaluation is purged walk-forward by calendar year so reported accuracy is
out-of-sample. The production fit is then refit on all labeled bars.

This is educational research, not a live execution engine or advice.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_DIR = os.path.dirname(__file__)
_MODEL_PATH = os.path.join(_DIR, ".pred_model.joblib")
_META_PATH = os.path.join(_DIR, ".pred_model_meta.json")
_CACHE_DIR = os.path.join(_DIR, ".pred_cache")

HORIZON = 5
EMBARGO = 10
WARMUP = 260
MAX_TRAIN_NAMES = 160
CONF_HI = 0.62
CONF_LO = 0.38
# Daily bars before decimalization / ETF era add noise vs. 2019–now OOS.
MIN_HISTORY = pd.Timestamp("2004-01-01")

FEATURES = [
    "rsi", "macd_hist_n", "bb_pos", "sma_gap",
    "mom_10", "mom_21", "mom_63", "mom_252",
    "ret_1", "ret_5", "ret_21",
    "vol_ratio", "atr_pct", "vol_20",
    "dist_high_20", "dist_low_20",
    "stoch_k", "cci", "adx", "di_gap", "trend_200",
]

_lock = threading.RLock()
_bundle: Optional[Dict[str, Any]] = None
_status: Dict[str, Any] = {"state": "idle", "message": None}


def training_universe() -> List[str]:
    from scanner import NIFTY50
    from markets import SP100_FALLBACK

    us = [s for s, _ in SP100_FALLBACK]
    names = list(dict.fromkeys(list(NIFTY50) + us))
    return names[:MAX_TRAIN_NAMES]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_meta(meta: Dict[str, Any]) -> None:
    try:
        tmp = _META_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f)
        os.replace(tmp, _META_PATH)
    except Exception:
        pass


def load_meta() -> Dict[str, Any]:
    try:
        if os.path.exists(_META_PATH):
            with open(_META_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _features_from_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    from scanner import (
        compute_rsi, compute_macd, compute_bollinger, compute_sma_crossover,
        compute_momentum, compute_volume_sma_ratio, compute_atr,
    )
    from indicators import compute_stochastic, compute_cci, compute_adx

    if df is None or df.empty or "close" not in df.columns:
        return pd.DataFrame()
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"] if "high" in df.columns else close, errors="coerce")
    low = pd.to_numeric(df["low"] if "low" in df.columns else close, errors="coerce")
    volume = pd.to_numeric(df["volume"] if "volume" in df.columns else 0.0, errors="coerce").fillna(0.0)

    rsi = compute_rsi(close, 14)
    _macd_line, _sig, hist = compute_macd(close)
    bb_u, bb_m, bb_l = compute_bollinger(close, 20)
    sma_s, sma_l = compute_sma_crossover(close, 10, 50)
    sma_200 = close.rolling(200, min_periods=60).mean()
    atr = compute_atr(high, low, close, 14)
    vol_r = compute_volume_sma_ratio(volume, 20)
    stoch_k, _ = compute_stochastic(high, low, close, 14)
    cci = compute_cci(high, low, close, 20)
    adx, pdi, mdi = compute_adx(high, low, close, 14)
    roll_h = high.rolling(20).max()
    roll_l = low.rolling(20).min()
    ret = close.pct_change()

    bw = (bb_u - bb_l).replace(0, np.nan)
    bb_pos = (2.0 * (close - bb_m) / bw).clip(-1, 1)
    bb_pos = bb_pos.where(close.notna() & bb_m.notna())

    macd_n = hist / (close.abs() + 1e-8)
    sma_gap = (sma_s - sma_l) / (close.abs() + 1e-8)
    out = pd.DataFrame({
        "rsi": rsi,
        "macd_hist_n": macd_n,
        "bb_pos": bb_pos,
        "sma_gap": sma_gap,
        "mom_10": compute_momentum(close, 10),
        "mom_21": compute_momentum(close, 21),
        "mom_63": compute_momentum(close, 63),
        "mom_252": compute_momentum(close, 252),
        "ret_1": ret,
        "ret_5": close.pct_change(5),
        "ret_21": close.pct_change(21),
        "vol_ratio": vol_r,
        "atr_pct": atr / (close.abs() + 1e-8),
        "vol_20": ret.rolling(20).std(),
        "dist_high_20": (close - roll_h) / (close.abs() + 1e-8),
        "dist_low_20": (close - roll_l) / (close.abs() + 1e-8),
        "stoch_k": stoch_k,
        "cci": cci,
        "adx": adx,
        "di_gap": (pdi - mdi),
        "trend_200": (close - sma_200) / (close.abs() + 1e-8),
    }, index=close.index)
    return out


def labeled_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    feats = _features_from_ohlc(df)
    if feats.empty or len(feats) < WARMUP + HORIZON + 5:
        return pd.DataFrame()
    close = pd.to_numeric(df["close"], errors="coerce")
    fwd = close.shift(-HORIZON) / close - 1.0
    feats = feats.iloc[WARMUP:]
    fwd = fwd.reindex(feats.index)
    out = feats.copy()
    out["fwd"] = fwd
    out["y"] = (fwd > 0).astype(float)
    out["ticker"] = ticker
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out["date"] = idx
    out = out.dropna(subset=["y", "fwd"])
    out = out[out["date"] >= MIN_HISTORY]
    return out


def _cache_path(sym: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in str(sym))
    return os.path.join(_CACHE_DIR, f"{safe}.pkl")


def _download_one(sym: str) -> Tuple[str, pd.DataFrame]:
    from ohlc import fetch_ohlcv

    path = _cache_path(sym)
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < 20 * 3600:
            try:
                cached = pd.read_pickle(path)
                if cached is not None and len(cached) > WARMUP + HORIZON:
                    return sym, cached
            except Exception:
                pass

    last = pd.DataFrame()
    for attempt in range(3):
        try:
            df, _ = fetch_ohlcv(sym, candle="24h", period="max")
        except Exception:
            df = pd.DataFrame()
        if df is not None and not df.empty:
            last = df
            try:
                os.makedirs(_CACHE_DIR, exist_ok=True)
                df.to_pickle(path)
            except Exception:
                pass
            return sym, df
        time.sleep(0.7 * (attempt + 1))
    return sym, last


def _build_panel(tickers: List[str], progress_cb=None) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows = []
    n = len(tickers)
    done = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_download_one, t): t for t in tickers}
        for fut in as_completed(futs):
            done += 1
            if progress_cb and (done % 8 == 0 or done == n):
                progress_cb(f"Downloaded {done}/{n}")
            if done % 8 == 0 or done == n:
                print(f"  downloaded {done}/{n}", flush=True)
            try:
                sym, df = fut.result()
            except Exception:
                continue
            if df is None or df.empty:
                continue
            part = labeled_frame(df, sym)
            if not part.empty:
                rows.append(part)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _fit(X: pd.DataFrame, y: np.ndarray):
    from sklearn.ensemble import HistGradientBoostingClassifier

    Xn = X.replace([np.inf, -np.inf], np.nan)
    clf = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.06,
        max_iter=180,
        l2_regularization=1.0,
        min_samples_leaf=120,
        max_bins=64,
        random_state=7,
    )
    clf.fit(Xn, y)
    return clf


def _metrics(y_true: np.ndarray, p_buy: np.ndarray, fwd: np.ndarray) -> Dict[str, Any]:
    pred = (p_buy >= 0.5).astype(int)
    n = int(len(y_true))
    if n == 0:
        return {"n": 0}
    acc = float(np.mean(pred == y_true))
    buy_mask = pred == 1
    sell_mask = pred == 0

    def _prec(mask, cls):
        if int(mask.sum()) == 0:
            return None
        return float(np.mean(y_true[mask] == cls))

    hi = (p_buy >= CONF_HI) | (p_buy <= CONF_LO)
    hi_acc = float(np.mean(pred[hi] == y_true[hi])) if int(hi.sum()) else None
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "buy_precision": None if _prec(buy_mask, 1) is None else round(_prec(buy_mask, 1), 4),
        "sell_precision": None if _prec(sell_mask, 0) is None else round(_prec(sell_mask, 0), 4),
        "buy_share": round(float(np.mean(pred)), 4),
        "actual_up_share": round(float(np.mean(y_true)), 4),
        "high_conf_n": int(hi.sum()),
        "high_conf_accuracy": None if hi_acc is None else round(hi_acc, 4),
        "high_conf_spread_pct": round(
            float((np.mean(fwd[hi & buy_mask]) - np.mean(fwd[hi & sell_mask])) * 100), 4
        ) if int((hi & buy_mask).sum()) and int((hi & sell_mask).sum()) else None,
        "mean_fwd_when_buy_pct": round(float(np.mean(fwd[buy_mask]) * 100), 4) if int(buy_mask.sum()) else None,
        "mean_fwd_when_sell_pct": round(float(np.mean(fwd[sell_mask]) * 100), 4) if int(sell_mask.sum()) else None,
        "spread_pct": round(
            float((np.mean(fwd[buy_mask]) - np.mean(fwd[sell_mask])) * 100), 4
        ) if int(buy_mask.sum()) and int(sell_mask.sum()) else None,
        "majority_baseline": round(float(max(np.mean(y_true), 1 - np.mean(y_true))), 4),
    }


def _walk_forward(panel: pd.DataFrame) -> Dict[str, Any]:
    years = list(range(2019, datetime.now(timezone.utc).year + 1))
    folds = []
    oos_y = []
    oos_p = []
    oos_f = []
    oos_mom = []
    Xall = panel[FEATURES]
    yall = panel["y"].astype(int).values
    fwd = panel["fwd"].astype(float).values
    mom = (panel["mom_21"].fillna(0).values >= 0).astype(int)
    dates = pd.to_datetime(panel["date"])

    for y0 in years:
        test_start = pd.Timestamp(f"{y0}-01-01")
        test_end = pd.Timestamp(f"{y0 + 1}-01-01")
        train_end = test_start - pd.Timedelta(days=EMBARGO)
        tr = dates < train_end
        te = (dates >= test_start) & (dates < test_end)
        if int(tr.sum()) < 4000 or int(te.sum()) < 400:
            continue
        print(f"  walk-forward {y0}: train={int(tr.sum())} test={int(te.sum())}", flush=True)
        clf = _fit(Xall.loc[tr], yall[tr.values])
        p = clf.predict_proba(Xall.loc[te].replace([np.inf, -np.inf], np.nan))[:, 1]
        yt = yall[te.values]
        ft = fwd[te.values]
        m = _metrics(yt, p, ft)
        m["year"] = y0
        m["momentum_baseline"] = round(float(np.mean(mom[te.values] == yt)), 4)
        folds.append(m)
        oos_y.append(yt)
        oos_p.append(p)
        oos_f.append(ft)
        oos_mom.append(mom[te.values])

    if oos_y:
        ycat = np.concatenate(oos_y)
        pooled = _metrics(ycat, np.concatenate(oos_p), np.concatenate(oos_f))
        pooled["momentum_baseline"] = round(float(np.mean(np.concatenate(oos_mom) == ycat)), 4)
    else:
        pooled = {"n": 0, "accuracy": None}
    return {"pooled": pooled, "folds": folds}


def train_and_save(tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    import joblib

    names = tickers or training_universe()
    with _lock:
        _status.update({"state": "running", "message": f"Fetching max daily history for {len(names)} names"})
    _save_meta({"state": "running", "message": _status["message"], "updated_at": _now_iso()})

    def prog(msg):
        with _lock:
            _status["message"] = msg
        _save_meta({**load_meta(), "state": "running", "message": msg, "updated_at": _now_iso()})

    try:
        panel = _build_panel(names, progress_cb=prog)
        if panel.empty:
            err = {"state": "error", "message": "No training rows — Yahoo returned empty history.", "updated_at": _now_iso()}
            _save_meta(err)
            with _lock:
                _status.update(err)
            return err

        # Cross-section label: beat that session's median 5-day return
        # within US or India. Absolute up/down is mostly market drift.
        panel["_mkt"] = np.where(panel["ticker"].astype(str).str.endswith(".NS"), "IN", "US")
        cnt = panel.groupby(["_mkt", "date"])["fwd"].transform("size")
        panel = panel.loc[cnt >= 15].copy()
        med = panel.groupby(["_mkt", "date"])["fwd"].transform("median")
        panel["y"] = (panel["fwd"] > med).astype(float)
        panel = panel.drop(columns=["_mkt"]).reset_index(drop=True)

        y = panel["y"].astype(int).values
        print(f"panel: {len(panel)} rows, {panel['ticker'].nunique()} names", flush=True)

        prog("Walk-forward evaluation…")
        wf = _walk_forward(panel)

        prog("Fitting production model…")
        clf = _fit(panel[FEATURES], y)
        joblib.dump({"model": clf, "features": FEATURES, "horizon": HORIZON}, _MODEL_PATH)

        names_used = sorted(panel["ticker"].astype(str).unique().tolist())
        date_min = str(pd.to_datetime(panel["date"]).min().date())
        date_max = str(pd.to_datetime(panel["date"]).max().date())
        pooled = wf.get("pooled") or {}
        meta = {
            "state": "ready",
            "message": None,
            "updated_at": _now_iso(),
            "horizon_bars": HORIZON,
            "names": names_used,
            "n_names": len(names_used),
            "n_rows": int(len(panel)),
            "date_min": date_min,
            "date_max": date_max,
            "features": FEATURES,
            "walk_forward": wf,
            "oos_accuracy": pooled.get("accuracy"),
            "oos_n": pooled.get("n"),
            "high_conf_accuracy": pooled.get("high_conf_accuracy"),
            "high_conf_n": pooled.get("high_conf_n"),
            "high_conf_spread_pct": pooled.get("high_conf_spread_pct"),
            "spread_pct": pooled.get("spread_pct"),
            "majority_baseline": pooled.get("majority_baseline"),
            "momentum_baseline": pooled.get("momentum_baseline"),
            "mean_fwd_when_buy_pct": pooled.get("mean_fwd_when_buy_pct"),
            "mean_fwd_when_sell_pct": pooled.get("mean_fwd_when_sell_pct"),
            "buy_precision": pooled.get("buy_precision"),
            "sell_precision": pooled.get("sell_precision"),
        }
        _save_meta(meta)
        with _lock:
            global _bundle
            _bundle = {"model": clf, "features": FEATURES, "horizon": HORIZON, "meta": meta}
            _status.update({"state": "ready", "message": None})
        return meta
    except Exception as e:
        err = {"state": "error", "message": str(e), "updated_at": _now_iso()}
        _save_meta(err)
        with _lock:
            _status.update(err)
        return err


def start_train_async(tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    with _lock:
        if _status.get("state") == "running":
            return get_report()
        _status.update({"state": "running", "message": "Starting"})
    t = threading.Thread(target=train_and_save, args=(tickers,), name="predi-train", daemon=True)
    t.start()
    return get_report()


def _load_bundle() -> Optional[Dict[str, Any]]:
    global _bundle
    with _lock:
        if _bundle and _bundle.get("model") is not None:
            return _bundle
    if not os.path.exists(_MODEL_PATH):
        return None
    try:
        import joblib
        raw = joblib.load(_MODEL_PATH)
        meta = load_meta()
        with _lock:
            _bundle = {**raw, "meta": meta}
        return _bundle
    except Exception:
        return None


def get_report() -> Dict[str, Any]:
    meta = load_meta()
    with _lock:
        st = dict(_status)
    if not meta:
        meta = {"state": st.get("state") or "idle", "message": st.get("message")}
    if st.get("state") == "running":
        meta["state"] = "running"
        meta["message"] = st.get("message") or meta.get("message")
    meta["available"] = bool(os.path.exists(_MODEL_PATH) and meta.get("state") == "ready")
    meta["max_train_names"] = MAX_TRAIN_NAMES
    meta["universe_size"] = len(training_universe())
    return meta


def predict_last(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    bundle = _load_bundle()
    if not bundle or bundle.get("model") is None:
        return None
    feats = _features_from_ohlc(df)
    if feats.empty:
        return None
    row = feats.iloc[[-1]][FEATURES].replace([np.inf, -np.inf], np.nan)
    try:
        p = float(bundle["model"].predict_proba(row)[0, 1])
    except Exception:
        return None
    if not np.isfinite(p):
        return None
    meta = bundle.get("meta") or load_meta()
    action = "BUY" if p >= 0.5 else "SELL"
    conf = abs(p - 0.5) * 2.0
    return {
        "p_buy": round(p, 4),
        "action": action,
        "confidence": round(conf, 4),
        "horizon": bundle.get("horizon") or HORIZON,
        "oos_accuracy": meta.get("oos_accuracy"),
        "high_conf_accuracy": meta.get("high_conf_accuracy"),
    }


if __name__ == "__main__":
    report = train_and_save()
    folds = (report.get("walk_forward") or {}).get("folds") or []
    print(json.dumps({
        **{k: report.get(k) for k in (
            "state", "n_names", "n_rows", "date_min", "date_max",
            "oos_accuracy", "high_conf_accuracy", "high_conf_n",
            "high_conf_spread_pct",
            "majority_baseline", "momentum_baseline", "spread_pct",
            "buy_precision", "sell_precision", "mean_fwd_when_buy_pct",
            "mean_fwd_when_sell_pct", "oos_n", "message",
        )},
        "folds": [{"year": f.get("year"), "accuracy": f.get("accuracy"), "n": f.get("n")} for f in folds],
    }, indent=2))
