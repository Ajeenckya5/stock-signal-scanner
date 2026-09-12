# Broadtape — 24/7 US + India insights desk

Named for the **broad tape**: the old ticker wire that printed every name, not just the leaders.

Two desks share the same research stack. Leave `python app.py` running and the **autopilot** thread keeps analyzing in the background: index pulse every 2 minutes, a full BUY/SELL book on your cadence, and 5-minute tape while NYSE or NSE is open. You can still choose the universe, candle, watchlist, and run a one-off scan.

| Desk | Candles | Markets | How to run |
|------|---------|---------|------------|
| **Insights desk** | 24h / daily, 1 month, 3 month, 6 month | US (NYSE/Nasdaq) and India (NSE) | `python app.py` → http://localhost:8000 |
| **Intraday** | 5-minute | Same | Same process: http://localhost:8000/intraday · or `python intraday_app.py` → http://localhost:8001 |

Signals are **BUY or SELL only** (no HOLD). This is research/educational software, not financial advice.

**Website:** [https://ajeenckya5.github.io/broadtape/](https://ajeenckya5.github.io/broadtape/) — the live desk (Overview, book, scans, 5-minute tape). It talks to the public API at `https://predi-stock.onrender.com`. First load can take a minute if that service was asleep.

---

## What the UI does

- **Overview** — live US / NSE clocks, index pulse (SPY, QQQ, IWM, Nifty, Bank Nifty), desk headline, alerts, conviction longs/shorts
- **Signal book** — autopilot results or your last manual scan; click a row for chart, stops, quant, news, 👍/👎
- **Choose & scan** — pick what runs 24/7 (universe, candle, cadence, optional watchlist, 5m-when-open) **or** fire a one-off scan with indicator checkboxes
- **Intraday 5m** — session VWAP, opening range, volume POC, RVOL, cumulative delta

Default autopilot universe is **liquid mix** (18 names, cap 60) so the desk stays fast. Do not leave S&P 500 on a 5-minute cadence.

---

## Long-term research stack

- Universes: liquid mix, Nifty 50 / Next 50, full India DB, US mega-caps, S&P 100, Nasdaq-100, S&P 500, or a Nifty 50 + US mega mix
- Candle construction: Yahoo daily bars for 24h; native monthly; calendar quarter and semi-annual bars resampled from daily
- Classical TA: RSI, MACD, Bollinger, SMA, stochastic, Williams %R, CCI, ADX, OBV, Stoch RSI, pivots, chart patterns, news sentiment, RLHF weights
- Quant overlay (see **Research stack** in the UI): Jegadeesh–Titman 12–1 momentum, short-term reversal, CAPM beta/alpha vs SPY or Nifty, BAB-style low-beta, Sharpe/Sortino/Calmar, historical VaR/CVaR, Parkinson / Garman–Klass / Yang–Zhang vol, RiskMetrics EWMA, GARCH(1,1), Hurst R/S, Lo–MacKinlay variance ratio, Dickey–Fuller, OU half-life, Amihud illiquidity, Kaufman ER, Kalman trend, Ichimoku, Supertrend, Keltner, Donchian, Chaikin MF, MFI, Heikin-Ashi, half-Kelly
- Cross-sectional ranks inside each scan (momentum, low-vol, Sharpe, relative strength)
- Deep dossier: `GET /research/{ticker}?candle=24h` (fundamentals when Yahoo provides them)

## Intraday 5-minute desk

Yahoo provides about 60 sessions of 5m bars. Session clocks: NSE 09:15–15:30 IST, US 09:30–16:00 ET.

- Opening-range breakout (Crabel, first 30 minutes)
- Session VWAP vs price
- Volume profile / POC / value area
- Relative volume by time of day
- Cumulative delta (tick-rule signed volume)
- Gap vs prior close, time-of-day seasonality
- 5m RSI/MACD, Hurst, Kalman, ATR stops

---

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open **http://localhost:8000** (insights desk) and **http://localhost:8000/intraday** (5-minute). Autopilot starts with the process.

Standalone 5-minute process:

```bash
python intraday_app.py
```

Open **http://localhost:8001**.

---

## Production on Render

1. Push to GitHub.
2. Render **Web Service**: build `pip install -r requirements-render.txt`, start `uvicorn app:app --host 0.0.0.0 --port $PORT`.
3. Optional: set `NEWSAPI_KEY` for `/train` and `/agent` (scanner uses Yahoo without it).
4. Heavy ML (torch / FinBERT) is local-only with full `requirements.txt`.

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Insights UI |
| `/intraday` | GET | 5-minute UI |
| `/desk/snapshot` | GET | Clocks, pulse, insights, autopilot book |
| `/desk/config` | POST | Universe, candle, cadence, watchlist, autopilot on/off |
| `/desk/run` | POST | Queue `full` \| `pulse` \| `long` \| `intraday` now |
| `/scan` | POST | Manual scan: `tickers`, `universe`, `candle` (`24h`\|`1mo`\|`3mo`\|`6mo`), `filter_action`, `indicators` |
| `/chart/{ticker}` | GET | Candles + patterns; query `candle=` |
| `/research/{ticker}` | GET | Full quant dossier |
| `/research/catalog` | GET | Paper / method list |
| `/search` | GET | US + India symbol search |
| `/train` `/agent` | POST | Optional news+MDN pipeline |

Intraday (mounted at `/intraday` or on port 8001): `POST /scan`, `GET /chart/{ticker}`, `GET /search`.

---

## Tech stack

Python · FastAPI · yfinance · pandas/numpy · Lightweight Charts · VADER news · optional PyTorch / FinBERT
