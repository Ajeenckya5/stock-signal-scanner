"""
RLHF - Reinforcement Learning from Human Feedback
=================================================
Collects user feedback on BUY/SELL signals and adapts factor weights
so the algorithm keeps improving.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

_STORE_PATH = os.path.join(os.path.dirname(__file__), ".rlhf_weights.json")

# Default factor weights. Must cover every id in scanner.ALL_SCORING_INDICATORS,
# otherwise feedback on those factors (stochastic, williams_r, cci, adx, obv,
# stoch_rsi) is silently dropped by record_feedback.
DEFAULT_WEIGHTS = {
    "rsi": 1.0,
    "macd": 1.0,
    "bollinger": 1.0,
    "sma": 1.0,
    "momentum": 1.0,
    "volume": 1.0,
    "stochastic": 1.0,
    "williams_r": 1.0,
    "cci": 1.0,
    "adx": 1.0,
    "obv": 1.0,
    "stoch_rsi": 1.0,
    "pattern": 1.0,
    "news": 1.0,
    "quant": 1.0,
    "ml": 1.15,
}

# Feedback history for adaptive learning
_FEEDBACK_PATH = os.path.join(os.path.dirname(__file__), ".rlhf_feedback.json")


def _load_weights() -> Dict[str, float]:
    if os.path.exists(_STORE_PATH):
        try:
            with open(_STORE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return dict(DEFAULT_WEIGHTS)


def _save_weights(w: Dict[str, float]) -> None:
    try:
        with open(_STORE_PATH, "w") as f:
            json.dump(w, f, indent=2)
    except Exception:
        pass


def _load_feedback() -> List[Dict]:
    if os.path.exists(_FEEDBACK_PATH):
        try:
            with open(_FEEDBACK_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_feedback(fb: List[Dict]) -> None:
    try:
        with open(_FEEDBACK_PATH, "w") as f:
            json.dump(fb[-5000:], f)  # keep last 5000
    except Exception:
        pass


def record_feedback(
    ticker: str,
    action: str,
    was_correct: bool,
    factors_present: Optional[List[str]] = None,
) -> Dict:
    """
    Record human feedback. Updates factor weights based on correctness.
    factors_present: which factors contributed (rsi, macd, bollinger, sma, momentum, volume, pattern, news)
    """
    weights = _load_weights()
    # Backfill any factor weights missing from an older saved file (e.g. extended
    # indicators added later) so feedback on them isn't silently dropped below.
    for k, v in DEFAULT_WEIGHTS.items():
        weights.setdefault(k, v)
    feedback_list = _load_feedback()

    factors = factors_present or list(weights.keys())
    delta = 0.05 if was_correct else -0.03  # reward correct, penalize incorrect

    for f in factors:
        if f in weights:
            weights[f] = max(0.2, min(2.0, weights[f] + delta))

    feedback_list.append({
        "ticker": ticker,
        "action": action,
        "was_correct": was_correct,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })
    _save_weights(weights)
    _save_feedback(feedback_list)

    return {"weights": weights, "feedback_count": len(feedback_list)}


def get_weights() -> Dict[str, float]:
    """Return current learned weights for use in scanner."""
    weights = _load_weights()
    for k, v in DEFAULT_WEIGHTS.items():
        weights.setdefault(k, v)
    return weights


def get_stats() -> Dict:
    """Return RLHF stats: feedback count, current weights, accuracy."""
    fb = _load_feedback()
    correct = sum(1 for x in fb if x.get("was_correct"))
    return {
        "feedback_count": len(fb),
        "accuracy": round(correct / len(fb), 3) if fb else 0,
        "weights": get_weights(),
    }


def reset_weights() -> Dict[str, float]:
    """Reset to default weights."""
    _save_weights(dict(DEFAULT_WEIGHTS))
    return get_weights()
